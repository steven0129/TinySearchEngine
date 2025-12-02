// Simple Node.js Express web server that provides a UI to interact with search.py API

const express = require('express');
const path = require('path');
const fetch = require('node-fetch');

const app = express();
const PORT = 3000;

// Serve static HTML UI
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Simple Search Engine UI</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 40px;
          background-color: #f4f4f4;
        }
        h1 {
          color: #333;
        }
        .container {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 0 10px rgba(0,0,0,0.1);
          max-width: 600px;
          margin: auto;
        }
        input, select, button {
          padding: 10px;
          margin: 5px 0;
          width: 100%;
          font-size: 16px;
        }
        button {
          background-color: #007BFF;
          color: white;
          border: none;
          cursor: pointer;
        }
        button:hover {
          background-color: #0056b3;
        }
        .results {
          margin-top: 20px;
        }
        .result-item {
          background: #e9ecef;
          padding: 10px;
          border-radius: 5px;
          margin-bottom: 10px;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Search Engine UI</h1>
        <label for="method">Search Method:</label>
        <select id="method">
          <option value="term">Term</option>
          <option value="tfidf">TF-IDF</option>
        </select>
        <label for="query">Enter your query:</label>
        <input type="text" id="query" placeholder="Type your search query..." />
        <button id="searchBtn">Search</button>
        <div class="results" id="results"></div>
      </div>

      <script>
        document.getElementById('searchBtn').addEventListener('click', async () => {
          const query = document.getElementById('query').value.trim();
          const method = document.getElementById('method').value;
          const resultsDiv = document.getElementById('results');
          resultsDiv.innerHTML = '<p>Searching...</p>';

          if (!query) {
            resultsDiv.innerHTML = '<p style="color:red;">Please enter a query.</p>';
            return;
          }

          try {
            const response = await fetch(\`http://localhost:5050/search?q=\${encodeURIComponent(query)}&method=\${method}\`);
            const data = await response.json();

            if (data.error) {
              resultsDiv.innerHTML = '<p style="color:red;">' + data.error + '</p>';
              return;
            }

            if (data.results && data.results.length > 0) {
              resultsDiv.innerHTML = '<h3>Results:</h3>' + data.results.map(id => 
                \`<div class="result-item">Document ID: \${id}</div>\`
              ).join('');
            } else {
              resultsDiv.innerHTML = '<p>No results found.</p>';
            }
          } catch (err) {
            resultsDiv.innerHTML = '<p style="color:red;">Error connecting to backend: ' + err.message + '</p>';
          }
        });
      </script>
    </body>
    </html>
  `);
});

app.listen(PORT, () => {
  console.log(`Web UI running at http://localhost:${PORT}`);
  console.log('Make sure search.py is running on port 5050.');
});