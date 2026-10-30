/**
 * SHADOW-only ASML/TSMC capture diagnostics.
 * Only bounded, predefined categories may reach the admin UI or logs.
 * No raw error messages, URLs, redirects, tokens, request bodies, or stack traces.
 */
export const ASML_CAPTURE_ERROR_LABELS = {
  SOURCE_SELECTION_INVALID: 'Le document sélectionné ne figure pas dans la liste autorisée.',
  SOURCE_ACCESS_DENIED: 'Le site émetteur refuse le téléchargement automatique (HTTP 401, 403 ou 451).',
  SOURCE_NOT_FOUND: 'Le document n’est pas disponible à cette adresse (HTTP 404 ou 410).',
  SOURCE_RATE_LIMITED: 'Le site émetteur limite temporairement les téléchargements (HTTP 429).',
  SOURCE_HTTP_FAILURE: 'Le serveur de l’émetteur a renvoyé une erreur HTTP.',
  SOURCE_REDIRECT_BLOCKED: 'Le document tente une redirection hors de la liste autorisée ; capture refusée.',
  SOURCE_RESPONSE_INVALID: 'La réponse ne contient pas un PDF conforme (format, type ou corps manquant).',
  SOURCE_SIZE_LIMIT: 'Le document dépasse la taille maximale admise pour cette source.',
  SOURCE_TIMEOUT: 'Le téléchargement a dépassé le délai de sécurité.',
  SOURCE_NETWORK_FAILURE: 'La connexion HTTPS à l’émetteur a échoué.',
  STORAGE_FAILURE: 'Le stockage privé SHADOW ou sa relecture SHA-256 a échoué.',
  SOURCE_POST_CUTOFF: 'La date de publication dépasse le cutoff autorisé.',
  CAPTURE_UNCLASSIFIED: 'La capture a échoué ; cause non déterminée par les contrôles disponibles.',
} as const;

export type AsmlCaptureErrorCode=keyof typeof ASML_CAPTURE_ERROR_LABELS;

export function classifyAsmlCaptureError(error:unknown):AsmlCaptureErrorCode {
  if(!(error instanceof Error))return 'CAPTURE_UNCLASSIFIED';
  const message=error.message;
  if(message==='ASML_V29_UNKNOWN_SOURCE')return 'SOURCE_SELECTION_INVALID';
  if(message==='ASML_V29_POST_CUTOFF_SOURCE')return 'SOURCE_POST_CUTOFF';
  const http=/^ASML_V29_ISSUER_FETCH_HTTP_([0-9]{3})$/.exec(message);
  if(http){
    const status=Number(http[1]);
    if([401,403,451].includes(status))return 'SOURCE_ACCESS_DENIED';
    if([404,410].includes(status))return 'SOURCE_NOT_FOUND';
    if(status===429)return 'SOURCE_RATE_LIMITED';
    return 'SOURCE_HTTP_FAILURE';
  }
  if(/^ASML_V29_(BAD_REDIRECT|REDIRECT_OUTSIDE_ALLOWLIST|REDIRECT_WITHOUT_LOCATION|TOO_MANY_REDIRECTS)$/.test(message))
    return 'SOURCE_REDIRECT_BLOCKED';
  if(/^ASML_V29_(UNEXPECTED_MIME|INVALID_PDF|EMPTY_RESPONSE_BODY)$/.test(message))
    return 'SOURCE_RESPONSE_INVALID';
  if(/^ASML_V29_(CONTENT_LENGTH_INVALID|BODY_EXCEEDS_LIMIT)$/.test(message))
    return 'SOURCE_SIZE_LIMIT';
  if(/^ASML_V29_PRIVATE_/.test(message))return 'STORAGE_FAILURE';
  if(['AbortError','TimeoutError'].includes(error.name))return 'SOURCE_TIMEOUT';
  if(error instanceof TypeError&&/fetch failed|network|fetch/i.test(message))
    return 'SOURCE_NETWORK_FAILURE';
  return 'CAPTURE_UNCLASSIFIED';
}

export function asmlCaptureErrorLabel(code:unknown):string {
  if(typeof code==='string'&&Object.prototype.hasOwnProperty.call(ASML_CAPTURE_ERROR_LABELS,code))
    return ASML_CAPTURE_ERROR_LABELS[code as AsmlCaptureErrorCode];
  return ASML_CAPTURE_ERROR_LABELS.CAPTURE_UNCLASSIFIED;
}
