// Read-only inventory. Requires the bundled @oai/artifact-tool runtime.
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
const [source, output] = process.argv.slice(2);
if (!source || !output) throw new Error('Usage: node clariti_table_inventory.mjs <data.xlsx> <output.json>');
const bytes = await fs.readFile(source);
const sourceSha = crypto.createHash('sha256').update(bytes).digest('hex');
if (sourceSha !== '2c0d086adb35fe383408055ed1ede2fd05c07e4bf6fa230782f99373bfb38a0f') throw new Error('Pinned XLSX digest mismatch');
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const overview = await wb.inspect({kind:'sheet',include:'id,name',maxChars:2000});
const sheet = wb.worksheets.getItem('Sheet1');
const rows = sheet.getUsedRange().values;
const headers = rows[0];
if (rows.length !== 501 || headers.length !== 28) throw new Error('Source shape changed: review before reusing this inventory');
const ididx=headers.indexOf('instance_id');
if (ididx < 0) throw new Error('Missing instance_id');
const findings={
  repository:'https://github.com/sani903/Teaching-Effective-Clarification',
  commit:'9d0f37e57dfde021e6589388f362d45b5901a356',
  source_file:'data/data.xlsx',source_sha256:sourceSha,
  code_sha256:crypto.createHash('sha256').update(await fs.readFile(new URL(import.meta.url))).digest('hex'),
  sheet:'Sheet1',range:'A1:AB501',data_rows:rows.length-1,
  columns:headers.map((name,j)=>({column:j+1,name,nonempty:rows.slice(1).filter(r=>r[j]!==null && r[j]!==undefined && r[j]!=='').length})),
  unique_instance_ids:new Set(rows.slice(1).map(r=>r[ididx])).size,
  raw_data_exported:false,workbook_modified:false,
  interpretation:'Task inputs, rewrites, gold patch/tests, and candidate questions. No column for observed clarification replies, execution trajectories, or per-run resolution. This does not exclude releases elsewhere.',
};
await fs.writeFile(output,JSON.stringify(findings,null,2)+'\n');
console.log(JSON.stringify({rows:findings.data_rows,columns:headers,unique_ids:findings.unique_instance_ids}));
