"""Start the single-user DocLens server with python server.py."""
import json
import os
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import models
from library import Library, MAX_BYTES, SUPPORTED

ROOT = Path(__file__).resolve().parent
STATIC = {'/':('index.html','text/html'), '/app.js':('app.js','text/javascript'),
          '/style.css':('style.css','text/css')}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def local(self):
        host = self.headers.get('Host','')
        return host in (f'localhost:{self.server.server_port}', f'127.0.0.1:{self.server.server_port}') and (
            self.headers.get('Origin') in (None,'http://'+host))

    def send(self, status, data, mime='application/json'):
        raw = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',mime+'; charset=utf-8')
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; frame-ancestors 'none'; object-src 'none'")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if not self.local():
            return self.send(403, {'error':'Только локальный доступ.'})
        path = self.path.split('?')[0]
        if path in STATIC:
            name, mime = STATIC[path]
            return self.send(200,(ROOT/'static'/name).read_bytes(),mime)
        if path == '/api/status':
            return self.send(200,{'models':models.availability(),'formats':sorted(SUPPORTED)})
        if path == '/api/documents':
            return self.send(200,self.server.library.documents())
        self.send(404,{'error':'Адрес не найден.'})

    def do_POST(self):
        if not self.local():
            return self.send(403, {'error':'Только локальный доступ.'})
        library = self.server.library
        try:
            size = int(self.headers.get('Content-Length',0))
            upload = self.path == '/api/import'
            if size < 0 or size > (MAX_BYTES if upload else 32768):
                return self.send(413, {'error':'Файл или запрос превышает лимит.'})
            if self.headers.get_content_type() != ('application/octet-stream' if upload else 'application/json'):
                return self.send(415, {'error':'Некорректный формат запроса.'})
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError('Передача файла прервалась.')
            if upload:
                count = library.replace(unquote(self.headers.get('X-Document-Path','')), raw)
                return self.send(200,{'count':count,'warning':library.index_vectors()})
            data = json.loads(raw)
            if not isinstance(data,dict):
                raise ValueError('Ожидается объект JSON.')
            if self.path == '/api/remove':
                library.remove(data.get('name'))
                return self.send(200,{'ok':True})
            if self.path not in ('/api/search','/api/answer'):
                return self.send(404,{'error':'Адрес не найден.'})
            question = data.get('question')
            if not isinstance(question,str) or not 1 <= len(question.strip()) <= 2000:
                raise ValueError('Введите вопрос длиной до 2000 символов.')
            if self.path == '/api/search':
                result = library.retrieve(question)
                return self.send(200,result if data.get('include_metadata') else result['sources'])
            ids = data.get('ids')
            if not isinstance(ids,list) or not 1 <= len(ids) <= 6 or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
                raise ValueError('Повторите поиск для выбора источников.')
            sources = library.selected(ids)
            self.send(200,{'claims':models.answer(data.get('provider'),question,sources)})
        except ValueError as exc:
            self.send(400, {'error':str(exc)})
        except urllib.error.HTTPError as exc:
            reasons = {401:'Проверьте API-ключ.',403:'Проверьте разрешения ключа и доступность региона.',
                       429:'Исчерпана квота. Повторите позже.',503:'Сервис временно недоступен.'}
            self.send(502,{'error':f'Сервис модели вернул HTTP {exc.code}. '+reasons.get(exc.code,'Проверьте настройки модели.')})
        except (OSError, TimeoutError):
            self.send(502,{'error':'Не удалось получить ответ сервиса. Проверьте соединение и состояние модели.'})
        except Exception:
            self.send(500,{'error':'Ошибка обработки. Проверьте доступ к локальной базе и повторите запрос.'})


if __name__ == '__main__':
    port = int(os.getenv('PORT', '8030'))
    db = Path(os.getenv('DOCLENS_DATA_DIR',ROOT/'data'))/'library.sqlite3'
    with ThreadingHTTPServer(('127.0.0.1',port),Handler) as server:
        server.library = Library(db)
        print(f'DocLens  http://127.0.0.1:{server.server_port}', flush=True)
        server.serve_forever()
