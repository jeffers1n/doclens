import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

from embeddings import Embeddings
from library import Library


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.library = Library(Path(folder.name)/'db')
        self.embedder = Mock()
        self.embedder.identity.return_value = 'nomic@test-v1'
        self.embedder.encode.side_effect = self.encode
        self.library.embedder = self.embedder

    @staticmethod
    def encode(texts, prefix, identity):
        if prefix == 'search_query: ':
            return [[1.0, 0.0] for _ in texts]
        return [[1.0, 0.0] if 'seventeen' in text else [0.0, 1.0] for text in texts]

    def documents(self):
        self.library.replace('backup.md', b'Backups are retained for seventeen days.')
        self.library.replace('help.md', b'The support office opens at nine.')

    def stored(self):
        with closing(self.library.connect()) as db:
            return db.execute('SELECT * FROM embeddings').fetchall()

    def test_semantic_match_without_shared_words_and_persistent_cache(self):
        self.documents()
        question = 'When does a saved copy expire?'
        with closing(self.library.connect()) as db:
            self.assertEqual(self.library._lexical(db,question),[])
        result = self.library.retrieve(question)
        self.assertEqual(result['mode'],'hybrid')
        self.assertIsNone(result['warning'])
        self.assertEqual(result['sources'][0]['name'],'backup.md')
        self.assertEqual(len(self.stored()),2)
        reopened = Library(self.library.path)
        reopened.embedder = self.embedder
        self.embedder.encode.reset_mock()
        reopened.retrieve(question)
        self.assertEqual([call.args[1] for call in self.embedder.encode.call_args_list],['search_query: '])

    def test_offline_fallback_covers_existing_documents(self):
        self.documents()
        self.embedder.identity.side_effect = OSError('offline')
        result = self.library.retrieve('seventeen')
        self.assertEqual(result['mode'],'bm25')
        self.assertTrue(result['warning'])
        self.assertEqual([s['name'] for s in result['sources']],['backup.md'])
        self.assertEqual(len(self.stored()),0)

    def test_exact_keyword_contributes_to_hybrid_ranking(self):
        self.documents()
        # The semantic stub prefers backup.md for every query. A keyword-only
        # match in help.md must still contribute to the combined ordering.
        result = self.library.retrieve('support')
        self.assertEqual(result['mode'],'hybrid')
        self.assertEqual(result['sources'][0]['name'],'help.md')

    def test_model_update_rebuilds_vectors(self):
        self.documents()
        self.assertIsNone(self.library.index_vectors())
        self.embedder.identity.return_value = 'nomic@test-v2'
        self.embedder.encode.reset_mock()
        self.library.retrieve('saved copy')
        self.assertTrue(all(row['model']=='nomic@test-v2' for row in self.stored()))
        self.assertEqual(self.embedder.encode.call_args_list[0].args[1],'search_document: ')

    def test_replace_and_delete_invalidate_old_vectors(self):
        self.documents()
        self.library.index_vectors()
        self.library.replace('backup.md',b'New manual.')
        self.library.remove('help.md')
        self.assertEqual(len(self.stored()),0)
        result = self.library.retrieve('manual')
        self.assertEqual(result['sources'][0]['text'],'New manual.')
        self.assertEqual(len(self.stored()),1)

    def test_batch_failure_is_retried_without_losing_text(self):
        for n in range(17):
            self.library.replace(f'file{n}.txt',b'seventeen')
        self.embedder.encode.side_effect = [[[1.0,0.0]]*16, OSError('offline')]
        self.assertTrue(self.library.index_vectors())
        self.assertEqual(len(self.stored()),16)
        self.assertEqual(len(self.library.documents()),17)
        self.embedder.encode.side_effect = self.encode
        self.embedder.encode.reset_mock()
        self.assertIsNone(self.library.index_vectors())
        self.assertEqual(len(self.stored()),17)
        self.assertEqual(len(self.embedder.encode.call_args.args[0]),1)

    def test_deleted_passage_is_not_reintroduced_by_indexer(self):
        self.documents()
        def concurrent_delete(texts, prefix, identity):
            self.library.remove('backup.md')
            return self.encode(texts,prefix,identity)
        self.embedder.encode.side_effect = concurrent_delete
        self.assertIsNone(self.library.index_vectors())
        self.assertEqual(len(self.stored()),1)
        self.assertEqual(self.library.documents()[0]['name'],'help.md')

    def test_legacy_library_migrates_without_reupload(self):
        # Emulate the exact tables that existed before vector search.
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)/'old.sqlite3'
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE passages(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT,location TEXT,text TEXT);
                CREATE VIRTUAL TABLE words USING fts5(terms);
                INSERT INTO passages VALUES(1,'old.md','fragment 1','seventeen days');
                INSERT INTO words(rowid,terms) VALUES(1,'seventeen days');
            """)
        migrated = Library(path)
        migrated.embedder = self.embedder
        self.assertEqual(migrated.retrieve('expiry')['mode'],'hybrid')
        self.assertEqual(migrated.selected([1])[0]['name'],'old.md')

    def test_wrong_vector_dimension_falls_back_to_lexical(self):
        self.documents()
        self.library.index_vectors()
        self.embedder.encode.return_value = None
        self.embedder.encode.side_effect = lambda *args: [[1.0,0.0,0.0]]
        result = self.library.retrieve('seventeen')
        self.assertEqual(result['mode'],'bm25')
        self.assertTrue(result['warning'])
        self.assertEqual(result['sources'][0]['name'],'backup.md')


class EmbeddingAdapterTests(unittest.TestCase):
    def test_normalization_and_task_prefix(self):
        model = Embeddings()
        with patch('embeddings.request',return_value={'embeddings':[[3.0,4.0]]}) as request, patch.object(model,'identity',return_value='same'):
            self.assertEqual(model.encode(['text'],'search_document: ','same'),[[0.6,0.8]])
        self.assertEqual(request.call_args.args[1]['input'],['search_document: text'])
        self.assertFalse(request.call_args.args[1]['truncate'])

    def test_changed_model_and_invalid_vectors_are_rejected(self):
        model = Embeddings()
        with patch('embeddings.request',return_value={'embeddings':[[1.0,0.0]]}), patch.object(model,'identity',return_value='changed'):
            with self.assertRaises(ValueError):
                model.encode(['text'],'search_query: ','old')
        with patch('embeddings.request',return_value={'embeddings':[[float('nan'),0.0]]}):
            with self.assertRaises(ValueError):
                model.encode(['text'],'search_query: ','same')
