#include <iostream>
#include <stdlib.h>
#include <malloc.h> // 包含 _aligned_malloc 函数
#include <immintrin.h> // AVX
#include <windows.h>
#include <xmmintrin.h> // SSE
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <omp.h>
#include <mpi.h>


using namespace std;

#define N 1000
#define ALIGNMENT 32 // 对齐方式为 16 字节
#define NUM_THREADS 8

float** m = NULL;
void m_reset() {
    if (m == NULL) {
        m = (float**)_aligned_malloc(sizeof(float*) * N, ALIGNMENT);
        for (int i = 0; i < N; i++) {
            m[i] = (float*)_aligned_malloc(sizeof(float) * N, ALIGNMENT);
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                m[i][j] = 1.0; // 设置对角线元素为1
            }
            else {
                m[i][j] = 0;   // 其他元素为0
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            m[i][j] = rand() % 100; // 生成0到99之间的随机数
        }
    }

    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                m[i][j] += m[k][j];
            }
        }
    }
}

void m_initAsEmpty() {
    m = new float* [N];
    for (int i = 0; i < N; i++) {
        m[i] = new float[N];
        memset(m[i], 0, N * sizeof(float));
    }

}


void freeUnalign() {
    for (int i = 0; i < N; i++) {
        _aligned_free(m[i]);
    }
    _aligned_free(m);
    m = NULL;
}


// 普通高斯消去法
void or_LU(float** m){
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }  
}

// 优化高斯消去法
// 对齐的SSE算法
void SIMD_LU(float** m) {
    for (int k = 0; k < N; k++) {
        __m256 vt = _mm256_set1_ps(m[k][k]); // AVX版本的载入单精度浮点数
        int j = k + 1;
        while ((int)(&m[k][j]) % 32) // AVX需要32字节对齐
        {
            m[k][j] = m[k][j] / m[k][k];
            j++;
        }
        for (; j + 8 <= N; j += 8) { // AVX版本加载/存储8个单精度浮点数
            __m256 va = _mm256_load_ps(&m[k][j]);
            va = _mm256_div_ps(va, vt);
            _mm256_store_ps(&m[k][j], va);
        }
        for (; j < N; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m256 vaik = _mm256_set1_ps(m[i][k]);
            j = k + 1;
            while ((int)(&m[k][j]) % 32) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
                j++;
            }
            for (; j + 8 <= N; j += 8) {
                __m256 vakj = _mm256_load_ps(&m[k][j]);
                __m256 vaij = _mm256_loadu_ps(&m[i][j]);
                __m256 vax = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vax);
                _mm256_storeu_ps(&m[i][j], vaij);
            }
            for (; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

// Pthread 动态线程
typedef struct {
    int k; //消去的轮次
    int t_id; //线程id
}threadParam_t1;

// 线程函数
DWORD WINAPI threadFunc1(LPVOID param) {
    threadParam_t1* p = reinterpret_cast<threadParam_t1*>(param);
    int k = p->k;        // 消去的轮次
    int t_id = p->t_id;  // 线程编号
    int i = k + t_id + 1;  // 获取自己的计算任务

    for (int j = k + 1; j < N; ++j) {
        m[i][j] -= m[i][k] * m[k][j];
    }
    m[i][k] = 0;

    return 0;
}

void pthread_LU() {
    for (int k = 0; k < N; ++k) {
        // 主线程做除法操作
        for (int j = k + 1; j < N; ++j) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;

        // 创建工作线程，进行消去操作
        int worker_count = N - 1 - k; // 工作线程数量
        vector<thread> threads(worker_count);
        vector<threadParam_t1> params(worker_count);

        // 分配任务
        for (int t_id = 0; t_id < worker_count; ++t_id) {
            params[t_id].k = k;
            params[t_id].t_id = t_id;
        }

        // 创建线程
        for (int t_id = 0; t_id < worker_count; ++t_id) {
            threads[t_id] = thread(threadFunc1, &params[t_id]);
        }

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; ++t_id) {
            threads[t_id].join();
        }
    }

}

// 静态线程 + 信号量同步版本
// 定义线程数据结构
typedef struct {
    int t_id; // 线程 id
} ThreadParam;

// 定义全局信号量
std::mutex mtx;
std::condition_variable cv_main;
std::condition_variable cv_workerstart[NUM_THREADS]; // 每个线程有自己专属的条件变量
std::condition_variable cv_workerend[NUM_THREADS];
bool ready = false;

// 线程函数
void threadFunc(ThreadParam* param, int n, float** A) {
    int t_id = param->t_id;

    for (int k = 0; k < n; ++k) {
        {
            std::unique_lock<std::mutex> lck(mtx);
            cv_workerstart[t_id].wait(lck, [&] {return ready; }); // 阻塞，等待主线程完成除法操作（操作自己专属的条件变量）
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            // 消去
            for (int j = k + 1; j < n; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }

        {
            std::unique_lock<std::mutex> lck(mtx);
            cv_main.notify_one(); // 唤醒主线程
            cv_workerend[t_id].wait(lck, [&] {return ready; }); // 阻塞，等待主线程唤醒进入下一轮
        }
    }
}

// omp高斯消去法
void omp_LU() {
    #pragma omp parallel if(parallel) num_threads(NUM_THREADS) private(i, j, k, tmp)
    {
        int tid = omp_get_thread_num(); // 获取线程id
        int i, j, k; // 声明私有变量
        float tmp; // 声明私有变量

        // 外循环
        for(k = 1; k < N; ++k){
            // 串行部分
            #pragma omp single
            {
                tmp = m[k][k];
                for(j = k + 1; j < N; ++j){
                    m[k][j] = m[k][j] / tmp;
                }
                m[k][k] = 1.0;
            }

            // 并行部分，使用行划分
            #pragma omp for
            for(i = k + 1; i < N; ++i){
                tmp = m[i][k];
                for(j = k + 1; j < N; ++j){
                    m[i][j] = m[i][j] - tmp * m[k][j];
                }
                m[i][k] = 0.0;
            }
            // 离开for循环时，各个线程默认同步，进入下一行的处理
        }
    }
}

// MPI
double mpi_LU(int argc, char* argv[]) {  //块划分
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);
    if (rank == 0) {  //0号进程初始化矩阵
        m_reset();

        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1是初始矩阵信息，向每个进程发送数据
            }
        }

    }
    else {
        m_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);  //此时每个进程都拿到了数据
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0;
            for (j = 0; j < total; j++) { //
                if (j != rank)
                    MPI_Send(&m[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0号消息表示除法完毕
            }
        }
        else {
            int src;
            if (k < N / total * total)//在可均分的任务量内
                src = k / (N / total);
            else
                src = total - 1;
            MPI_Recv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == 0) {//0号进程中存有最终结果
        end_time = MPI_Wtime();
    }
    MPI_Finalize();
    return end_time - start_time;
}

double async_mpi_LU(int argc, char* argv[]) {  //非阻塞通信
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {  //0号进程初始化矩阵
        m_reset();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //等待传递

    }
    else {
        m_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank];  //非阻塞传递
            for (j = rank + 1; j < total; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                MPI_Isend(&m[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; //若执行完自身的任务，可直接跳出
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
    }
    MPI_Finalize();
    return end_time - start_time;
}

double LU_mpi_async_omp(int argc, char* argv[]) {  //非阻塞通信+OpenMP
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    cout << MPI_Wtick();
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {  //0号进程初始化矩阵
        m_reset();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //等待传递

    }
    else {
        m_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            if ((begin <= k && k < end)) {
                for (j = k + 1; j < N; j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }
                m[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];  //非阻塞传递
                for (j = 0; j < total; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                    MPI_Isend(&m[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < N / total * total)//在可均分的任务量内
                    src = k / (N / total);
                else
                    src = total - 1;
                MPI_Request request;
                MPI_Irecv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
#pragma omp for schedule(guided)  //开始多线程
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
    }
    MPI_Finalize();
    return end_time - start_time;
}

double LU_mpi_async_avx(int argc, char* argv[]) {  //非阻塞通信+avx
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    //cout << MPI_Wtick();
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {  //0号进程初始化矩阵
        m_reset();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //等待传递

    }
    else {
        m_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        {
            if ((begin <= k && k < end)) {
                __m256 t1 = _mm256_set1_ps(m[k][k]);
                for (j = k + 1; j + 8 <= N; j += 8) {
                    __m256 t2 = _mm256_loadu_ps(&m[k][j]);  //AVX优化除法部分
                    t2 = _mm256_div_ps(t2, t1);
                    _mm256_storeu_ps(&m[k][j], t2);
                }
                for (; j < N; j++) {
                    m[k][j] = m[k][j] / m[k][k];
                }
                m[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];  //非阻塞传递
                for (j = 0; j < total; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                    MPI_Isend(&m[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < N / total * total)//在可均分的任务量内
                    src = k / (N / total);
                else
                    src = total - 1;
                MPI_Request request;
                MPI_Irecv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
        for (i = max(begin, k + 1); i < end; i++) {
            __m256 vik = _mm256_set1_ps(m[i][k]);   //AVX优化消去部分
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&m[k][j]);
                __m256 vij = _mm256_loadu_ps(&m[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&m[i][j], vij);
            }
            for (; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
    }
    MPI_Finalize();
    return end_time - start_time;
}



int main(int argc, char* argv[]) {
    LARGE_INTEGER frequency;        // 声明一个LARGE_INTEGER类型的变量来存储频率
    LARGE_INTEGER start, end;       // 声明两个LARGE_INTEGER类型的变量来存储开始和结束的计数值
    double elapsedTime;             // 声明一个double类型的变量来存储经过的时间
    // 获取计时器的频率
    QueryPerformanceFrequency(&frequency);
    //------------------------------------------------------------------------------
    // 普通高斯消去法
    m_reset();
    // 记录开始时间
    QueryPerformanceCounter(&start);
    or_LU(m);
    // 记录结束时间
    QueryPerformanceCounter(&end);
    // 计算经过的时间
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "普通高斯消去法时间: " << elapsedTime << " ms." << std::endl;



    //------------------------------------------------------------------------------
    // 优化高斯消去法部分
    

    //------------------------------------------------------------------------------
    // SIMD优化
    m_reset();
    // 记录开始时间
    QueryPerformanceCounter(&start);

    SIMD_LU(m);


    // 记录结束时间
    QueryPerformanceCounter(&end);
    // 计算经过的时间
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "SIMD优化高斯消去法时间: " << elapsedTime << " ms." << std::endl;


    //------------------------------------------------------------------------------
    // 静态线程优化
    m_reset();
    // 记录开始时间
    QueryPerformanceCounter(&start);
   
    // 创建线程
    std::vector<std::thread> threads;
    std::vector<ThreadParam> params(NUM_THREADS);
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        threads.push_back(std::thread(threadFunc, &params[t_id], N, std::ref(m)));
    }

    for (int k = 0; k < N; ++k) {
        // 主线程做除法操作
        for (int j = k + 1; j < N; ++j) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;

        // 开始唤醒工作线程
        {
            std::unique_lock<std::mutex> lck(mtx);
            ready = true;
            cv_workerstart[0].notify_all();
        }

        // 主线程睡眠（等待所有的工作线程完成此轮消去任务）
        {
            std::unique_lock<std::mutex> lck(mtx);
            cv_main.wait(lck, [&] {return ready; });
        }

        // 主线程再次唤醒工作线程进入下一轮次的消去任务
        {
            std::unique_lock<std::mutex> lck(mtx);
            ready = true;
            cv_workerend[0].notify_all();
        }
    }

    // 等待所有线程结束
    for (auto& thread : threads) {
        thread.join();
    }

    // 记录结束时间
    QueryPerformanceCounter(&end);
    // 计算经过的时间
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "静态线程优化高斯消去法时间: " << elapsedTime << " ms." << std::endl;

    //------------------------------------------------------------------------------
    //// 动态线程优化
    //m_reset();
    //// 记录开始时间
    //QueryPerformanceCounter(&start);

    //pthread_LU();


    //// 记录结束时间
    //QueryPerformanceCounter(&end);
    //// 计算经过的时间
    //elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    //cout << "动态线程优化高斯消去法时间: " << elapsedTime << " ms." << std::endl;

    //------------------------------------------------------------------------------
    // omp优化
    m_reset();
    // 记录开始时间
    QueryPerformanceCounter(&start);

    omp_LU();


    // 记录结束时间
    QueryPerformanceCounter(&end);
    // 计算经过的时间
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "omp优化高斯消去法时间: " << elapsedTime << " ms." << std::endl;

    //------------------------------------------------------------------------------
    //MPI
    elapsedTime = LU_mpi_async_avx(argc, argv); // 调用 LU_mpi_async_avx 函数
    cout << "MPI优化高斯消去法时间: " << elapsedTime * 1000.0 << " ms." << std::endl;


    // 释放内存
    freeUnalign();
    system("pause");
    return 0;
    
}
