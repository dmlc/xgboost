extern "C" {

void* allocate(size_t nbyte);
void deallocate(void* ptr, size_t nbyte);
void set_log_callback(void (*callback) (const char*));

}
