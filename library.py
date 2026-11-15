"""Import documents and search a local SQLite full-text index."""
import io
import re
import sqlite3
import struct
import threading
import zipfile
from contextlib import closing
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

import snowballstemmer
from pypdf import PdfReader

from embeddings import Embeddings

SUPPORTED = {'.txt', '.md', '.markdown', '.rst', '.html', '.htm', '.pdf', '.docx',
             '.csv', '.json', '.yaml', '.yml', '.toml', '.adoc'}
MAX_BYTES = 12 * 1024 * 1024
MAX_CHARS = 2_000_000
EXCLUDED = {'node_modules', '__pycache__', 'venv', 'vendor'}
STOP = set('и в во к на по о от из для как что где ли это или у с со а the an a for of and or to how do i is are in on'.split())
local = threading.local()


def document_path(name):
    if not isinstance(name, str) or len(name) > 1024:
        raise ValueError('Некорректное имя файла.')
    parts = name.replace('\\', '/').split('/')
    if any(not p or p.startswith('.') or p in EXCLUDED or ':' in p or
           any(ord(c) < 32 for c in p) for p in parts):
        raise ValueError('Скрытые и служебные пути пропускаются.')
    path = PurePosixPath(*parts)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError('Формат файла не поддерживается.')
    return str(path)


class HTMLText(HTMLParser):
    HEAD_ELEMENTS = {'base', 'link', 'meta', 'title', 'noscript', 'script', 'style', 'template', 'noframes'}
    HIDDEN_ELEMENTS = {'title', 'script', 'style', 'template', 'noscript', 'noframes'}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.in_head = False
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in self.HIDDEN_ELEMENTS:
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag == 'head':
            self.in_head = True
        elif self.in_head and tag not in self.HEAD_ELEMENTS and tag != 'html':
            # HTML permits omitted head/body tags. Body content ends the head.
            self.in_head = False
        if tag in ('p', 'br', 'li', 'div', 'section', 'tr', 'td', 'th', 'h1', 'h2', 'h3', 'pre'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag == 'head':
            self.in_head = False
        if tag in ('p', 'li', 'div', 'tr', 'td', 'th', 'pre'):
            self.parts.append('\n')

    def handle_data(self, data):
        if self.hidden:
            return
        if self.in_head:
            if not data.strip(' \t\n\r\f'):
                return
            self.in_head = False
        self.parts.append(data)


def extract(name, raw):
    suffix = PurePosixPath(name).suffix.lower()
    stream = io.BytesIO(raw)
    if suffix == '.pdf':
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ValueError('PDF защищён паролем.')
        for n, page in enumerate(reader.pages, 1):
            yield f'стр. {n}', page.extract_text() or ''
    elif suffix == '.docx':
        with zipfile.ZipFile(stream) as doc:
            info = doc.getinfo('word/document.xml')
            if info.file_size > 16 * 1024 * 1024:
                raise ValueError('Слишком большой распакованный DOCX.')
            tree = ElementTree.fromstring(doc.read(info))
            word = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            paragraphs = [''.join(t.text or '' for t in p.iter(word+'t'))
                          for p in tree.iter(word+'p')]
            yield '', '\n'.join(paragraphs)
    else:
        text = raw.decode('utf-8-sig')
        if '\0' in text:
            raise ValueError('Файл содержит двоичные данные.')
        if suffix in ('.html', '.htm'):
            parser = HTMLText()
            parser.feed(text)
            parser.close()
            text = ''.join(parser.parts)
        yield '', text


@lru_cache(maxsize=32768)
def stem(word):
    language = 'russian' if re.search('[а-яё]', word) else 'english'
    if not hasattr(local, 'stemmers'):
        local.stemmers = {}
    if language not in local.stemmers:
        local.stemmers[language] = snowballstemmer.stemmer(language)
    return local.stemmers[language].stemWord(word)


def terms(text):
    return [stem(word) for word in re.findall(r'[^\W_]+', text.lower())
            if word not in STOP and len(word) < 100]


def passages(name, raw):
    if len(raw) > MAX_BYTES:
        raise ValueError('Максимальный размер файла — 12 Mb.')
    total = 0
    rows = []
    try:
        for page, text in extract(name, raw):
            total += len(text)
            if total > MAX_CHARS:
                raise ValueError('Документ превышает 2 млн символов.')
            text = text.strip()
            offset = 0
            while offset < len(text):
                end = min(len(text), offset + 1500)
                if end < len(text):
                    cut = max(text.rfind('\n', offset+800, end), text.rfind(' ', offset+800, end))
                    if cut >= 0:
                        end = cut
                fragment = text[offset:end].strip()
                if fragment:
                    location = f'фрагмент {len(rows)+1}'
                    rows.append((name, f'{page}, {location}' if page else location,
                                 fragment, ' '.join(terms(name+' '+fragment))))
                if end == len(text):
                    break
                offset = end - 140
    except UnicodeDecodeError:
        raise ValueError('Сохраните текстовый файл в UTF-8.') from None
    except ValueError:
        raise
    except Exception:
        raise ValueError('Не удалось прочитать документ. Проверьте формат и целостность файла.') from None
    if not rows:
        raise ValueError('В документе нет текста. Для скана сначала выполните OCR.')
    return rows


class Library:
    def __init__(self, path, vector_search=True):
        self.path = Path(path)
        self.embedder = Embeddings() if vector_search else None
        self.embedding_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS passages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, location TEXT, text TEXT);
                CREATE INDEX IF NOT EXISTS doc_name ON passages(name);
                CREATE VIRTUAL TABLE IF NOT EXISTS words USING fts5(terms);
                CREATE TABLE IF NOT EXISTS embeddings(
                    passage_id INTEGER PRIMARY KEY REFERENCES passages(id) ON DELETE CASCADE,
                    model TEXT NOT NULL, vector BLOB NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.execute('PRAGMA foreign_keys=ON')
        db.row_factory = sqlite3.Row
        return db

    def documents(self):
        with closing(self.connect()) as db:
            return [dict(r) for r in db.execute(
                'SELECT name,count(*) AS count FROM passages GROUP BY name ORDER BY name')]

    def replace(self, name, raw):
        name = document_path(name)
        rows = passages(name, raw)
        with closing(self.connect()) as db, db:
            self._remove(db, name)
            for name, location, text, tokens in rows:
                row = db.execute('INSERT INTO passages(name,location,text) VALUES(?,?,?)',
                                 (name, location, text))
                db.execute('INSERT INTO words(rowid,terms) VALUES(?,?)',(row.lastrowid,tokens))
        return len(rows)

    @staticmethod
    def _remove(db, name):
        db.execute('DELETE FROM embeddings WHERE passage_id IN (SELECT id FROM passages WHERE name=?)',(name,))
        db.execute('DELETE FROM words WHERE rowid IN (SELECT id FROM passages WHERE name=?)',(name,))
        db.execute('DELETE FROM passages WHERE name=?',(name,))

    def remove(self, name):
        with closing(self.connect()) as db, db:
            self._remove(db, document_path(name))

    def _lexical(self, db, question):
        tokens = list(dict.fromkeys(terms(question)))[:64]
        if not tokens:
            return []
        expression = ' OR '.join('"'+word+'"' for word in tokens)
        return [row['id'] for row in db.execute("""
            SELECT p.id FROM words JOIN passages p ON p.id=words.rowid
            WHERE words MATCH ? ORDER BY bm25(words),p.id LIMIT 24""", (expression,))]

    def _index_vectors(self, identity):
        # HTTP calls run outside write transactions. Each committed batch can be
        # retried, and deleted/replaced passage IDs are never reintroduced.
        with closing(self.connect()) as db:
            pending = db.execute("""
                SELECT p.id,p.name,p.text FROM passages p LEFT JOIN embeddings e ON e.passage_id=p.id
                WHERE e.passage_id IS NULL OR e.model != ? ORDER BY p.id""",(identity,)).fetchall()
        for start in range(0, len(pending), 16):
            batch = pending[start:start+16]
            vectors = self.embedder.encode(
                [row['name'][:160]+'\n'+row['text'] for row in batch], 'search_document: ', identity)
            with closing(self.connect()) as db, db:
                for row, vector in zip(batch, vectors):
                    blob = struct.pack('<'+str(len(vector))+'f', *vector)
                    db.execute("""
                        INSERT OR REPLACE INTO embeddings(passage_id,model,vector)
                        SELECT id,?,? FROM passages WHERE id=?""", (identity,blob,row['id']))

    def index_vectors(self):
        if self.embedder is None:
            return 'Поиск по словам. Векторный поиск отключён.'
        try:
            with self.embedding_lock:
                self._index_vectors(self.embedder.identity())
        except (OSError, ValueError, KeyError, TypeError):
            return 'Поиск по словам. Для поиска по смыслу запустите Ollama с nomic-embed-text.'
        return None

    def retrieve(self, question):
        warning = None
        identity = None
        query = None
        if self.embedder is not None:
            try:
                with self.embedding_lock:
                    identity = self.embedder.identity()
                    self._index_vectors(identity)
                    query = self.embedder.encode([question], 'search_query: ', identity)[0]
            except (OSError, ValueError, KeyError, TypeError):
                warning = 'Поиск по словам. Для поиска по смыслу запустите Ollama с nomic-embed-text.'
                query = None
        else:
            warning = 'Поиск по словам. Векторный поиск отключён.'
        with closing(self.connect()) as db:
            # One read snapshot keeps passages and vectors aligned during imports.
            db.execute('BEGIN')
            lexical = self._lexical(db, question)
            passages = {row['id']:dict(row) for row in db.execute('SELECT * FROM passages')}
            semantic = []
            if query is not None:
                stored = db.execute('SELECT passage_id,vector FROM embeddings WHERE model=?',(identity,)).fetchall()
                if len(stored) != len(passages):
                    warning = 'Поиск по словам. Библиотека обновилась, повторите поиск для индексации.'
                    query = None
                else:
                    for row in stored:
                        blob = row['vector']
                        if len(blob) != len(query)*4:
                            warning = 'Поиск по словам. Размеры embeddings не совпали, обновите индекс.'
                            query = None
                            break
                        vector = struct.unpack('<'+str(len(query))+'f', blob)
                        semantic.append((row['passage_id'], sum(a*b for a,b in zip(query, vector))))
            rankings = [lexical]
            if query is not None:
                rankings.append([i for i,score in sorted(semantic,key=lambda pair:(-pair[1],pair[0]))[:24]])
            scores = {}
            for ranking in rankings:
                for rank, passage_id in enumerate(ranking, 1):
                    scores[passage_id] = scores.get(passage_id,0) + 1/(60+rank)
            selected = sorted(scores,key=lambda i:(-scores[i],i))[:6]
            return {'sources':[passages[i] for i in selected],
                    'mode':'hybrid' if query is not None else 'bm25', 'warning':warning}

    def search(self, question):
        return self.retrieve(question)['sources']

    def selected(self, ids):
        with closing(self.connect()) as db:
            rows = db.execute('SELECT id,name,location,text FROM passages WHERE id IN ('+
                              ','.join('?' for _ in ids)+')',ids)
            by_id = {row['id']:dict(row) for row in rows}
        if set(by_id) != set(ids):
            raise ValueError('Документ изменился. Повторите поиск.')
        return [by_id[i] for i in ids]
