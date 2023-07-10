extern void goLogCallbackFunc(char* msg);

void bridge_log_callback(const char* msg) {
  goLogCallbackFunc((char*)msg);
}
