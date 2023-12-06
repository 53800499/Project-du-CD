
import http.server

port = 800
adresse = ("", port)

server = http.server.HTTPServer

handler = http.server.CGIHTTPRequestHandler
handler.cgi_directories = ["/"]  # Répertoire où se trouvent vos scripts CGI

httpd = server(adresse, handler)

print(f"Serveur démarré sur le port {port}")
httpd.serve_forever()
