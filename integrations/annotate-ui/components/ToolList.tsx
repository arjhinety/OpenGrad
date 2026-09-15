type Json = Record<string, unknown>;

function asObject(value: unknown): Json | null {
  if (typeof value === "string") {
    try {
      return asObject(JSON.parse(value));
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Json) : null;
}

function describeType(schema: Json): string {
  const type = schema.type;
  if (Array.isArray(type)) return type.join(" | ");
  if (typeof type === "string") {
    const items = asObject(schema.items);
    return items && typeof items.type === "string" ? `${type}<${items.type}>` : type;
  }
  if (Array.isArray(schema.enum)) return "enum";
  return "—";
}

function format(value: unknown): string {
  if (value === undefined) return "";
  return typeof value === "string" ? value : JSON.stringify(value);
}

function Tool({ raw }: { raw: unknown }) {
  const wrapper = asObject(raw);
  if (!wrapper) {
    return (
      <div className="tool">
        <pre className="json">{format(raw)}</pre>
      </div>
    );
  }
  // OpenAI-style {type: "function", function: {...}} and flat {name, parameters} are both common.
  const tool = asObject(wrapper.function) ?? wrapper;
  const parameters = asObject(tool.parameters) ?? {};
  const properties = asObject(parameters.properties) ?? {};
  const required = new Set<string>(
    [parameters.required, tool.required, wrapper.required].flatMap((list) =>
      Array.isArray(list) ? list.map(String) : [],
    ),
  );
  const names = Object.keys(properties);
  return (
    <div className="tool">
      <header>
        <div className="name">{String(tool.name ?? "(unnamed tool)")}</div>
        {tool.description ? <div className="desc">{String(tool.description)}</div> : null}
      </header>
      {names.length ? (
        <div className="params-scroll">
          <table className="params">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Type</th>
                <th>Default</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {names.map((name) => {
                const schema = asObject(properties[name]) ?? {};
                return (
                  <tr key={name}>
                    <td className="pname">
                      {name}
                      {required.has(name) ? <span className="req" title="required"> *</span> : null}
                    </td>
                    <td className="ptype">{describeType(schema)}</td>
                    <td className="ptype">{format(schema.default)}</td>
                    <td>
                      {format(schema.description)}
                      {Array.isArray(schema.enum) ? ` (one of: ${schema.enum.map(format).join(", ")})` : ""}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="params-scroll">
          <table className="params">
            <tbody>
              <tr>
                <td className="empty">No parameters declared.</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
      <details>
        <summary>Raw schema</summary>
        <pre>{JSON.stringify(raw, null, 2)}</pre>
      </details>
    </div>
  );
}

export function ToolList({ tools }: { tools: unknown }) {
  const list = Array.isArray(tools) ? tools : tools ? [tools] : [];
  if (!list.length) return <div className="body empty">No tools were offered to the assistant.</div>;
  return (
    <div className="tools">
      <div className="empty" style={{ fontSize: 12.5 }}>
        {list.length} tool{list.length === 1 ? "" : "s"} offered · <span className="req">*</span> required parameter
      </div>
      {list.map((tool, index) => (
        <Tool key={index} raw={tool} />
      ))}
    </div>
  );
}
