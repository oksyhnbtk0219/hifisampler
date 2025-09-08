import logging
from pathlib import Path
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from concurrent.futures import ThreadPoolExecutor

from backend.resampler import Resampler

server_ready = False

def split_arguments(input_string):
    otherargs = input_string.split(' ')[-11:]
    file_path_strings = ' '.join(input_string.split(' ')[:-11])

    first_file, second_file = file_path_strings.split('.wav ')
    return [first_file+".wav", second_file] + otherargs


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if server_ready:
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Server Ready')
            logging.info("Responded 200 OK to readiness check.")
        else:
            self.send_response(503)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Server Initializing')
            logging.info(
                "Responded 503 Service Unavailable to readiness check (server not ready).")
        return

    def do_POST(self):
        if not server_ready:
            logging.warning(
                "Received POST request before server was fully ready. Sending 503.")
            self.send_response(503)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Server initializing, please retry.')
            return
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data_string = post_data.decode('utf-8')
        logging.info(f"post_data_string: {post_data_string}")
        try:
            sliced = split_arguments(post_data_string)
            in_file_path = Path(sliced[0])
            out_file_path = Path(sliced[1])
            note_info_for_log = f"'{in_file_path.stem}' -> '{out_file_path.name}'"
            logging.info(f"Processing {note_info_for_log} begins...")

            # === Execute Resampler within try...except ===
            Resampler(*sliced)
            # If Resampler completes without exception, it's considered successful *by the server*
            logging.info(f"Processing {note_info_for_log} successful.")
            self.send_response(200)  # Send 200 OK
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Success: {note_info_for_log}".encode('utf-8'))

        except FileNotFoundError:
            error_msg = f"Error processing {note_info_for_log}: Input file not found."
            logging.error(error_msg, exc_info=True)  # Log full traceback
            self.send_response(404)  # Not Found
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(
                f"{error_msg}\n{traceback.format_exc()}".encode('utf-8'))

        except Exception:
            # Catch any other exception during Resampler execution
            error_msg = f"[Error processing {note_info_for_log}: An internal error occurred."
            # Log the full traceback for debugging
            logging.error(error_msg, exc_info=True)
            self.send_response(500)  # Internal Server Error
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            # Send error details back (optional, consider security if sensitive info might leak)
            self.wfile.write(
                f"{error_msg}\n{traceback.format_exc()}".encode('utf-8'))

        '''
        except Exception as e:
            trcbk = traceback.format_exc()
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"An error occurred.\n{trcbk}".encode('utf-8'))
        self.send_response(200)
        self.end_headers()
        '''


class ThreadPoolHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, max_workers):
        super().__init__(server_address, RequestHandlerClass)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_request(self, request, client_address):
        self.executor.submit(self.process_request_thread,
                             request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def run(server_class=ThreadPoolHTTPServer, handler_class=RequestHandler, port=8572, max_workers=1):
    global server_ready

    server_address = ('', port)
    httpd = server_class(server_address, handler_class, max_workers=max_workers)

    logging.info(f'Listening on port {server_address[1]} with {max_workers} worker threads...')
    server_ready = True
    httpd.serve_forever()