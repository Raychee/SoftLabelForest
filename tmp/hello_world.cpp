# include <iostream>
# include <omp.h>
# include <unistd.h>

int main(int argc, const char** argv) {
    clock_t t1 = clock();
    # pragma omp parallel for num_threads(3)
    for(unsigned i = 0; i < 10; ++i) {
        std::cout << "Thread " << omp_get_thread_num()
                  << " of " << omp_get_num_threads() << " is sleeping.\n";
        sleep(1);
    }
    clock_t t2 = clock();
    std::cout << "Total time = " << t2-t1 << std::endl;
    for(unsigned i = 0; i < 10; ++i) {
         std::cout << "Thread " << omp_get_thread_num()
                  << " of " << omp_get_num_threads() << " is sleeping.\n";
        sleep(1);
    }
    clock_t t3 = clock();
    std::cout << "Total time = " << t3-t2 << std::endl;
    return 0;
}
