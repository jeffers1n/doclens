"""Local Nomic embeddings, normalized for cosine similarity."""
import math

from models import OLLAMA, request

MODEL = 'nomic-embed-text:latest'


class Embeddings:
    def identity(self):
        tags = request(OLLAMA+'/api/tags', timeout=3)
        for model in tags.get('models', []):
            if model.get('name') == MODEL and not model.get('remote_model'):
                return MODEL+'@'+model['digest']
        raise ValueError('В Ollama отсутствует '+MODEL)

    def encode(self, texts, prefix, identity):
        result = request(OLLAMA+'/api/embed', {
            'model':MODEL, 'input':[prefix+text for text in texts], 'truncate':False,
        }, timeout=90)
        vectors = result['embeddings']
        if len(vectors) != len(texts) or not vectors:
            raise ValueError('Неполный ответ модели embeddings')
        dimension = len(vectors[0])
        normalized = []
        for vector in vectors:
            if not dimension or len(vector) != dimension or not all(math.isfinite(v) for v in vector):
                raise ValueError('Некорректные embeddings')
            length = math.hypot(*vector)
            if not length:
                raise ValueError('Пустой embedding')
            normalized.append([value/length for value in vector])
        if self.identity() != identity:
            raise ValueError('Модель embeddings изменилась во время индексации')
        return normalized
