FROM node:22-alpine
ENV NODE_ENV=production
WORKDIR /srv/d29b
COPY --chown=node:node services/d29b-cloud-run-signer/policy.mjs ./policy.mjs
COPY --chown=node:node services/d29b-cloud-run-signer/kms.mjs ./kms.mjs
COPY --chown=node:node services/d29b-cloud-run-signer/server.mjs ./server.mjs
USER node
EXPOSE 8080
CMD ["node","server.mjs"]
