let resolutions;
let sources;
let modulePrefixes;

export function initialize(data) {
  resolutions = new Map(data.resolutions);
  sources = new Map(data.sources);
  modulePrefixes = data.modulePrefixes;
}

export async function resolve(specifier, context, nextResolve) {
  const url = resolutions.get(specifier);
  if (url !== undefined) return { url, shortCircuit: true };

  const resolved = await nextResolve(specifier, context);
  if (
    resolved.format === 'commonjs'
    && modulePrefixes.some((prefix) => resolved.url.startsWith(prefix))
  ) {
    return { ...resolved, format: 'module' };
  }
  return resolved;
}

export async function load(url, context, nextLoad) {
  const source = sources.get(url);
  if (source !== undefined) return { format: 'module', source, shortCircuit: true };
  return nextLoad(url, context);
}
