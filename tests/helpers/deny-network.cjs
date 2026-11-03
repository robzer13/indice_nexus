// Preload with NODE_OPTIONS=--require=./tests/helpers/deny-network.cjs.
// Defense in depth for local qualification, alongside the execution sandbox.
// This monkeypatch is not a general network isolation boundary (native code,
// subprocesses and references captured before preloading are outside its scope).
const deny = () => { throw new Error('LOCAL_QUALIFICATION_NETWORK_DENIED'); };
const net = require('node:net');
const connect = net.Socket.prototype.connect;
net.Socket.prototype.connect = function (...args) {
  // tsx uses a Windows named pipe for parent/child IPC. It is not a network socket.
  const options = Array.isArray(args[0]) ? args[0][0] : args[0];
  const path = typeof options === 'string' ? options : options?.path;
  if (typeof path === 'string' &&
      (path.startsWith('\\\\.\\pipe\\') || path.startsWith('\\\\?\\pipe\\'))) {
    return connect.apply(this, args);
  }
  return deny();
};
require('node:tls').connect = deny;
require('node:dgram').Socket.prototype.send = deny;
for (const protocol of ['node:http', 'node:https']) {
  const api = require(protocol);
  api.request = deny;
  api.get = deny;
}
const dns = require('node:dns');
const dnsPromises = require('node:dns/promises');
const dnsMethods = [
  'lookup', 'lookupService', 'resolve', 'resolve4', 'resolve6', 'resolveAny',
  'resolveCaa', 'resolveCname', 'resolveMx', 'resolveNaptr', 'resolveNs',
  'resolvePtr', 'resolveSoa', 'resolveSrv', 'resolveTlsa', 'resolveTxt', 'reverse',
];
for (const api of [dns, dns.Resolver.prototype]) {
  for (const method of dnsMethods) {
    if (typeof api[method] === 'function') api[method] = deny;
  }
}
for (const api of [dnsPromises, dnsPromises.Resolver.prototype]) {
  for (const method of dnsMethods) {
    if (typeof api[method] === 'function') api[method] = async () => deny();
  }
}
globalThis.fetch = async () => deny();
require('node:module').syncBuiltinESMExports();
