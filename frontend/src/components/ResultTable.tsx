export function ResultTable({ observation }: { observation: Record<string, unknown> }) {
  const columns = Array.isArray(observation.columns) ? observation.columns.map(String) : [];
  const rows = Array.isArray(observation.rows) ? (observation.rows as unknown[][]) : [];
  if (!columns.length) return null;
  return (
    <div className="result-wrap">
      <table className="result-table">
        <thead><tr>{columns.map((column, index) => <th key={`${column}-${index}`}>{column}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>{row.map((cell, columnIndex) => <td key={columnIndex}>{cell === null ? <i>NULL</i> : String(cell)}</td>)}</tr>
          ))}
        </tbody>
      </table>
      <div className="result-foot">已显示 {rows.length} 行{Boolean(observation.truncated) && " · 结果已截断"}</div>
    </div>
  );
}
