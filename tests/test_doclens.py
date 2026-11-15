import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from library import Library, document_path
import models


class DocumentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.library = Library(Path(tmp.name)/'db', vector_search=False)

    def test_import_replacement_and_deletion(self):
        for folder, text in [('a','расписание занятий'), ('b','архивные копии')]:
            self.library.replace(folder+'/guide.md',text.encode())
        found = self.library.search('архивная копия')
        self.assertEqual([s['name'] for s in found],['b/guide.md'])
        self.library.replace('b/guide.md','другая тема'.encode())
        with self.assertRaises(ValueError):
            self.library.selected([found[0]['id']])
        self.assertEqual(self.library.search('архив'),[])
        self.library.remove('b/guide.md')
        self.assertEqual(len(self.library.documents()),1)

    def test_html_and_damaged_replacement(self):
        self.library.replace('a.html', b'<head><title>secret</title></head><p>visible</p><script>secret</script>')
        with self.assertRaises(ValueError):
            self.library.replace('a.html',b'\xff')
        self.assertEqual(len(self.library.search('visible')),1)
        self.assertEqual(self.library.search('secret'),[])

    def test_docx_and_pdf_locations(self):
        doc = io.BytesIO()
        with zipfile.ZipFile(doc,'w') as archive:
            archive.writestr('word/document.xml','<document xmlns="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><p><r><t>guide</t></r></p></document>')
        self.library.replace('a.docx',doc.getvalue())
        pdf = PdfWriter()
        pdf.add_blank_page(200,200)
        page = pdf.add_blank_page(200,200)
        font = DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 10 20 Td (example) Tj ET')
        page[NameObject('/Contents')] = pdf._add_object(stream)
        blob = io.BytesIO()
        pdf.write(blob)
        self.library.replace('a.pdf',blob.getvalue())
        self.assertEqual(self.library.search('example')[0]['location'],'стр. 2, фрагмент 1')
        self.assertEqual(self.library.search('guide')[0]['name'],'a.docx')

    def test_reject_paths(self):
        for path in ('../notes.md','.env','docs/.git/test.md','/abs.md','notes.exe'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                document_path(path)

    def test_html_tables_keep_terms_separate(self):
        self.library.replace('table.html',b'<table><tr><td>Retention</td><td>17 days</td></tr></table>')
        self.assertEqual(len(self.library.search('Retention')),1)

    def test_html_without_head_end_tag(self):
        self.library.replace('guide.html',b'<!doctype html><html><head><title>Hidden</title><body><p>Retention 17 days.</p></body></html>')
        self.assertEqual(len(self.library.search('Retention')),1)
        self.assertEqual(self.library.search('Hidden'),[])

    def test_html_without_head_end_or_body_start_tags(self):
        documents = [
            '<!doctype html><html><head><title>Manual</title><h1>Manual</h1><p>Retention 17 days.</p></html>',
            '<html><head><title>Hidden</title>Retention 17 days.</html>',
            '<head><title>Hidden</title>Retention 17 days.',
            '<title>Hidden</title><p>Retention 17 days.</p>',
        ]
        for document in documents:
            with self.subTest(document=document):
                self.library.replace('guide.html',document.encode())
                self.assertEqual(len(self.library.search('Retention')),1)
                self.assertEqual(self.library.search('Hidden'),[])

    def test_html_metadata_and_scripts_stay_out_of_document_text(self):
        head = '''<head>
            <title>Hidden title</title>
            <meta name="description" content="Hidden metadata"/>
            <link rel="stylesheet" href="hidden.css"/>
            <style>hidden {color: red}</style>
            <script>const hidden = "<body>hidden</body>";</script>
            <template><div>hidden</div><template>hidden</template></template>
        '''
        for body in ('<body><p>Retention 17 days.</p>', '<p>Retention 17 days.</p>', 'Retention 17 days.'):
            with self.subTest(body=body):
                document = head+body+'<script>hidden</script><style>hidden {}</style></html>'
                self.library.replace('guide.html',document.encode())
                self.assertEqual(len(self.library.search('Retention')),1)
                self.assertEqual(self.library.search('Hidden'),[])

    def test_html_fragment_preserves_table_text_after_hidden_content(self):
        self.library.replace('fragment.html',b'<title>Hidden</title><script>hidden</script><style>hidden {}</style><table><tr><th>Retention</th><td>17 days</td></tr></table>')
        found = self.library.search('Retention')
        self.assertEqual(len(found),1)
        self.assertIn('17 days',found[0]['text'])
        self.assertEqual(self.library.search('Hidden'),[])


class ModelTests(unittest.TestCase):
    sources = [{'name':'a.md','text':'17 days'}]

    def test_gemini_rest_request_and_key(self):
        response = {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(
            {'claims':[{'answer':'17 дней','source':'S1'}]})}]}}]}
        with patch.dict('os.environ',{'GEMINI_API_KEY':'demo','GEMINI_FREE_TIER_CONFIRMED':'yes'}), patch('models.request',return_value=response) as call:
            self.assertEqual(models.answer('gemini','Как долго?',self.sources)[0]['source'],'S1')
        url, body, headers = call.call_args.args
        self.assertNotIn('demo',url)
        self.assertNotIn('demo',json.dumps(body))
        self.assertEqual(headers['x-goog-api-key'],'demo')
        self.assertIn('17 days',body['contents'][0]['parts'][0]['text'])

    def test_qwen_rejects_unknown_source(self):
        with patch('models.local_qwen',return_value='qwen2.5:1.5b'), patch('models.request',return_value={
            'message':{'content':'{"claims":[{"answer":"ok","source":"S9"}]}'}}), self.assertRaises(ValueError):
            models.answer('qwen','question',self.sources)

    def test_no_implicit_cloud(self):
        with patch.dict('os.environ',{},clear=True), patch('models.request') as call, self.assertRaises(ValueError):
            try:
                models.answer('gemini','question',self.sources)
            finally:
                call.assert_not_called()
