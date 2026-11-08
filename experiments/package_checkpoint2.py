"""Build a narrow submission archive; never includes unrelated workspace files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/checkpoint2_2026-10-27'
ZIP=ROOT/'outputs/checkpoint2_2026-10-27.zip'

START='''# Чекпойнт 2: разведочный анализ данных

Губарева Анна Владимировна, 27 октября 2026.

1. Откройте `output/pdf/checkpoint2_eda_2026-10-27.pdf`: основной отчёт, 7 страниц.
2. `outputs/checkpoint2_2026-10-27/CHECKLIST.md` сопоставляет материалы с четырьмя критериями README.
3. `outputs/checkpoint2_2026-10-27/DATA_CARD.md` описывает первую версию данных; JSONL находится в соседней папке data/.
4. Пять графиков PNG/SVG и `analysis.json` находятся там же. Редактируемый текст отчёта - REPORT.md.

## Повторить анализ без API

Из корня этой распакованной папки, при установленных NumPy и Matplotlib:

```bash
python3 experiments/checkpoint2_eda.py --output outputs/repeated_eda
```

Сборка использует включённые аналитические входы E002 и не скачивает модели. Версии проверенного окружения указаны в requirements-eda.txt. Можно создать отдельное окружение и установить их обычным способом, если библиотек нет.

Проверка внешних raw-данных требует их отдельного получения; команда приведена в REPRODUCE.md. Сырые внешние каталоги, пользовательские профили и языковые модели сюда не включены. Сведения о лицензии и ограничения производной таблицы приведены в DATA_CARD.md.

Повторная генерация самого PDF необязательна: нужен ReportLab, Pillow и DejaVu Sans (через CHECKPOINT_FONT_DIR или системные шрифты). Команда: `python3 experiments/build_checkpoint2_report.py`. Готовый проверенный PDF уже включён.

Исследовательское приложение содержит E002/E004, новый эксперимент E005, аудит внешних сред и краткий итог 27 октября. Часть ссылок ведёт к другим материалам полного исследовательского проекта. Для воспроизведения EDA приложения не нужны. Наличие анализа не означает подтверждённую новизну или выполненную внешнюю агентную проверку.

`PACKAGE_MANIFEST.json` содержит SHA256 всех файлов архива. Архив собран по критериям README, без загрузки в учебную систему.
'''


def main():
    paths=[ROOT/'README.md',ROOT/'output/pdf/checkpoint2_eda_2026-10-27.pdf']
    paths+=sorted(p for p in OUT.rglob('*') if p.is_file())
    paths += [ROOT/'experiments'/f for f in ['checkpoint2_eda.py','build_checkpoint2_report.py','learn_question_value.py','question_value_diagnostics.py','package_checkpoint2.py','limited_feedback_pilot.py']]
    for folder in ['learn_question_value','question_value_diagnostics','limited_feedback_pilot','external_environment_audit']:
        paths+=sorted(p for p in (ROOT/'results'/folder).rglob('*') if p.is_file())
    for name in ['E002_PROTOCOL.md','E002_RESULTS.md','E004_PROTOCOL.md','E004_RESULTS.md','LITERATURE_UPDATE_2026-10-20.md','STAGE1_REPORT.md','STAGE1_SEARCH_LOG.md','RESEARCH_CANDIDATES.md','E005_PROTOCOL.md','E005_RESULTS.md','EXTERNAL_ENVIRONMENT_AUDIT_2026-10-27.md','UPDATE_2026-10-27.md']:
        paths.append(ROOT/'research'/name)
    assert all(p.is_file() for p in paths)
    entries={str(p.relative_to(ROOT)):p.read_bytes() for p in paths}
    entries['START_HERE.md']=START.encode()
    entries['requirements-eda.txt']=b'numpy==2.0.2\nmatplotlib==3.9.4\n'
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}
    entries['PACKAGE_MANIFEST.json']=json.dumps({'date':'2026-10-27','sha256':hashes},indent=2,ensure_ascii=False).encode()
    with zipfile.ZipFile(ZIP,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo('checkpoint2_2026-10-27/'+name,date_time=(2026,10,27,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    with zipfile.ZipFile(ZIP) as z:assert z.testzip() is None
    print(json.dumps({'archive':str(ZIP),'files':len(entries),'bytes':ZIP.stat().st_size},ensure_ascii=False))


if __name__=='__main__':main()
