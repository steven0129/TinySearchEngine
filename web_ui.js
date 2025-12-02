// Simple Node.js Express web server that provides a UI to interact with search.py API

const express = require('express');
const path = require('path');
const fetch = require('node-fetch');

const app = express();
const PORT = 3000;

app.use(express.static('public'));
app.listen(PORT, () => {
  console.log(`Web UI running at http://localhost:${PORT}`);
  console.log('Make sure search.py is running on port 5050.');
});