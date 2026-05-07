#!/usr/bin/env bash
# 퍼머링크를 /%postname%/ 로 바꾸고 .htaccess 에 표준 WP rewrite 블록을 기록.
# /wp-json/... 형태의 REST API 라우팅이 동작하려면 1회 실행 필요.
# (docker compose down -v 로 볼륨 초기화 후에도 다시 실행해야 함)
set -euo pipefail

WP_CONTAINER="${WP_CONTAINER:-llm-jacky-wordpress-1}"
NETWORK="${NETWORK:-llm-jacky_default}"

docker run --rm \
  --network "$NETWORK" \
  --volumes-from "$WP_CONTAINER" \
  -u 33:33 \
  -e WORDPRESS_DB_HOST=db:3306 \
  -e WORDPRESS_DB_USER=wordpress \
  -e WORDPRESS_DB_PASSWORD=wordpress \
  -e WORDPRESS_DB_NAME=wordpress \
  wordpress:cli option update permalink_structure '/%postname%/'

docker exec -u root -i "$WP_CONTAINER" bash -c 'cat > /var/www/html/.htaccess << "HTACCESS"
# BEGIN WordPress
RewriteEngine On
RewriteBase /
RewriteRule ^index\.php$ - [L]
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule . /index.php [L]
# END WordPress
HTACCESS
chown www-data:www-data /var/www/html/.htaccess'

echo "✓ permalinks=/%postname%/, .htaccess written"
echo -n "  /wp-json/ → HTTP "
curl -sS -o /dev/null -w "%{http_code}\n" http://localhost:8080/wp-json/
