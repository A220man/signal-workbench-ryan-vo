FROM node:24-slim AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . ./
RUN npm run build

FROM nginx:alpine
RUN chmod -R 777 /var/cache/nginx /var/run /var/log/nginx && \
    sed -i 's/listen       80;/listen       80;/' /etc/nginx/conf.d/default.conf && \
    touch /var/run/nginx.pid && \
    chmod 777 /var/run/nginx.pid
COPY --from=builder /app/dist /usr/share/nginx/html
USER 10001:10001
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=5s --start-period=3s --retries=3 \
  CMD wget -q --spider http://127.0.0.1:80/ || exit 1
CMD ["nginx", "-g", "daemon off;"]
