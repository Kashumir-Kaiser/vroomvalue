# Browser E2E

The MVP browser acceptance flow is: open `/`, submit the default Toyota Camry example, confirm a USD estimate plus an 80% range and three factors, then submit feedback through the API and confirm the admin metrics increment. A Playwright harness is a Phase-2 improvement; the current clean-system gate includes a manual browser pass after the automated API integration suite.