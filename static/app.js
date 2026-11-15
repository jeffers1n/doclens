'use strict';
const $ = id => document.getElementById(id);
const node = (tag, text) => {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  return el;
};
let config;
let searching = false;
let importing = false;
let docCount = 0;

async function api(path, options = {}) {
  const response = await fetch('/api/' + path, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.error || 'Ошибка запроса.');
  return data;
}
const post = (path, data) => api(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
function controls() {
  $('go').disabled = !config || !docCount || searching || importing;
  for (const el of [$('filesButton'), $('folderButton'), $('mode'), ...document.querySelectorAll('#library button')]) {
    el.disabled = !config || searching || importing;
  }
}
function resetResults() {
  $('sources').replaceChildren();
  $('answers').replaceChildren();
  $('status').textContent = '';
}
async function refresh() {
  const docs = await api('documents');
  docCount = docs.length;
  $('fileCount').textContent = docCount;
  $('emptyState').hidden = docCount > 0;
  $('library').replaceChildren();
  for (const doc of docs) {
    const row = node('li');
    const remove = node('button','×');
    remove.className = 'light';
    remove.setAttribute('aria-label','Удалить из библиотеки ' + doc.name);
    remove.onclick = async () => {
      if (!confirm('Удалить из индекса «' + doc.name + '»?')) return;
      try {
        importing = true;
        controls();
        await post('remove',{name:doc.name});
        resetResults();
        await refresh();
      } catch (e) { $('importStatus').textContent = e.message; }
      finally { importing = false; controls(); }
    };
    const name = node('span',doc.name);
    name.title = doc.name;
    row.append(name, remove);
    $('library').append(row);
  }
  controls();
}
function modeInfo() {
  const mode = $('mode').value;
  const notes = [];
  if (mode === 'gemini' || mode === 'both') {
    notes.push('Gemini получит вопрос, имена найденных файлов и выбранные поиском выдержки.');
    if (config.models.gemini.issue) notes.push(config.models.gemini.issue);
  }
  if (mode === 'qwen' || mode === 'both') notes.push(config.models.qwen.issue || ('На этом компьютере · ' + config.models.qwen.model));
  $('modelInfo').textContent = notes.join(' ');
  $('modelInfo').hidden = !notes.length;
}
async function importFiles(input) {
  const files = Array.from(input.files);
  input.value = '';
  if (!files.length) return;
  importing = true;
  controls();
  resetResults();
  const problems = $('problems').querySelector('ul');
  problems.replaceChildren();
  $('problems').hidden = true;
  let added = 0;
  let skipped = 0;
  let searchWarning = '';
  try {
    for (const [index, file] of files.entries()) {
      const name = file.webkitRelativePath || file.name;
      $('importStatus').textContent = 'Загрузка и индексация ' + (index+1) + ' из ' + files.length;
      try {
        if (name.split('/').some(part => part.startsWith('.') || ['node_modules','venv','__pycache__','vendor'].includes(part))) throw Error('служебный путь');
        if (!config.formats.some(ext => name.toLowerCase().endsWith(ext))) throw Error('неподдерживаемый формат');
        if (file.size > 12 * 1024 * 1024) throw Error('больше 12 МиБ');
        const result = await api('import',{method:'POST', headers:{'Content-Type':'application/octet-stream','X-Document-Path':encodeURIComponent(name)},body:file});
        if (result.warning) searchWarning = result.warning;
        added++;
      } catch (e) {
        skipped++;
        if (skipped <= 100) problems.append(node('li',name + ' — ' + e.message));
      }
    }
    $('problems').hidden = !skipped;
    $('importStatus').textContent = 'Добавлено или обновлено: ' + added + '. Пропущено: ' + skipped + '.';
    if (searchWarning) $('importStatus').textContent += ' ' + searchWarning;
    await refresh();
  } catch (e) { $('importStatus').textContent = e.message; }
  finally { importing = false; controls(); }
}
function showSources(sources) {
  $('sources').append(node('h2','Найденные фрагменты'));
  sources.forEach((source,index) => {
    const card = node('article');
    card.id = 'S' + (index+1);
    const heading = node('div');
    heading.className = 'source-heading';
    const reference = node('span',card.id);
    reference.className = 'source-reference';
    heading.append(reference,node('h3',source.name));
    card.append(heading,node('small',source.location),node('blockquote',source.text));
    $('sources').append(card);
  });
}
async function modelAnswer(provider,question,sources) {
  const box = node('article');
  box.className = 'answer-card';
  box.append(node('h3',config.models[provider].model));
  const state = node('p','Готовит ответ…');
  box.append(state);
  $('answers').append(box);
  try {
    const {claims} = await post('answer',{provider,question,ids:sources.map(s=>s.id)});
    state.textContent = claims.length ? '' : 'Модель не нашла в выдержках ответа.';
    for (const claim of claims) {
      const link = node('a','[' + claim.source + ']');
      link.href = '#' + claim.source;
      const p = node('p',claim.answer + ' ');
      p.append(link);
      box.append(p);
    }
  } catch (e) { state.textContent = e.message; }
}
$('form').onsubmit = async event => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question || searching || importing) return;
  searching = true;
  controls();
  resetResults();
  $('status').textContent = 'Ищу в документах…';
  const mode = $('mode').value;
  try {
    const result = await post('search',{question,include_metadata:true});
    const sources = Array.isArray(result) ? result : result.sources;
    const searchStatus = Array.isArray(result)
      ? 'Поиск по словам. Перезапустите сервер, чтобы включить обновлённый поиск.'
      : (result.warning || 'Поиск по смыслу и словам.');
    $('status').textContent = sources.length ? searchStatus : searchStatus + ' Совпадений нет. Попробуйте другую формулировку.';
    if (sources.length) {
      showSources(sources);
      if (mode !== 'search') {
        const providers = mode === 'both' ? ['qwen','gemini'] : [mode];
        await Promise.all(providers.map(p=>modelAnswer(p,question,sources)));
      }
    }
  } catch (e) { $('status').textContent = e.message; }
  finally { searching = false; controls(); }
};
for (const id of ['files','folder']) {
  $(id+'Button').onclick = ()=>$(id).click();
  $(id).onchange = ()=>importFiles($(id));
}
$('mode').onchange = modeInfo;
async function init() {
  controls();
  try {
    config = await api('status');
    $('files').accept = config.formats.join(',');
    $('types').textContent = 'PDF, DOCX, Markdown, HTML и текстовые файлы. До 12 МиБ на файл.';
    $('mode').value = config.models.qwen.issue ? 'search' : 'qwen';
    await refresh();
    modeInfo();
  } catch (e) { $('status').textContent = e.message; }
}
init();
