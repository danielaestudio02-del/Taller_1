// Genera informe/informe.pdf a partir de informe/informe.html (Chromium de Playwright).
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.resolve(__dirname, '../informe/informe.html'));
  await page.pdf({ path: path.resolve(__dirname, '../informe/informe.pdf'), format: 'Letter', printBackground: true,
                   margin: { top: '1.6cm', bottom: '1.6cm', left: '1.7cm', right: '1.7cm' } });
  await browser.close();
})();
