import React, { useState } from 'react';
import './App.css';

interface QueryResponse {
  query: string;
  answer: string;
  results: Array<{
    content: string;
    metadata: Record<string, any>;
    similarity_score: number;
  }>;
}

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:8000/api/v1/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data: QueryResponse = await res.json();
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG Pipeline</h1>
        <p>Ask questions about your documents</p>
      </header>

      <main>
        <form onSubmit={handleSubmit} className="query-form">
          <div className="input-group">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your question..."
              disabled={loading}
            />
            <button type="submit" disabled={loading || !query.trim()}>
              {loading ? 'Asking...' : 'Ask'}
            </button>
          </div>
        </form>

        {error && <div className="error">{error}</div>}

        {response && (
          <div className="response">
            <h2>Answer</h2>
            <p>{response.answer}</p>

            <h3>Sources</h3>
            <div className="sources">
              {response.results.map((result, index) => (
                <div key={index} className="source">
                  <p><strong>Similarity:</strong> {result.similarity_score.toFixed(3)}</p>
                  <p>{result.content}</p>
                  <details>
                    <summary>Metadata</summary>
                    <pre>{JSON.stringify(result.metadata, null, 2)}</pre>
                  </details>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;