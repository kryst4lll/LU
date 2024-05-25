#include <iostream>
#include <stdlib.h>
#include <malloc.h> // 包含 _aligned_malloc 函数
#include <immintrin.h>
#include <windows.h>
#include <xmmintrin.h>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <omp.h>

using namespace std;

#define N 2000
#define ALIGNMENT 16 // 对齐方式为 16 字节
#define NUM_THREADS 7

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
        __m128 vt = _mm_set1_ps(m[k][k]);
        int j = k + 1;
        while ((int)(&m[k][j]) % 16)
        {
            m[k][j] = m[k][j] / m[k][k];
            j++;
        }
        for (; j + 4 <= N; j += 4) {
            __m128 va = _mm_load_ps(&m[k][j]);   // 将四个单精度浮点数从内存加载到向量寄存器
            va = _mm_div_ps(va, vt);    // 这里是向量对位相除
            _mm_store_ps(&m[k][j], va); // 将四个单精度浮点数从向量寄存器存储到内存
        }
        for (; j < N; j++) {
            m[k][j] = m[k][j] / m[k][k];    // 该行结尾处有几个元素还未计算
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m128 vaik = _mm_set1_ps(m[i][k]);
            j = k + 1;
            while ((int)(&m[k][j]) % 16)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
                j++;
            }
            for (; j + 4 <= N; j += 4) {
                __m128 vakj = _mm_load_ps(&m[k][j]); // 将四个单精度浮点数从内存加载到向量寄存器
                __m128 vaij = _mm_loadu_ps(&m[i][j]);    // 将四个单精度浮点数从内存加载到向量寄存器
                __m128 vax = _mm_mul_ps(vakj, vaik);   // 这里是向量对位相乘
                vaij = _mm_sub_ps(vaij, vax);  // 这里是向量对位相减
                _mm_storeu_ps(&m[i][j], vaij);   // 将四个单精度浮点数从向量寄存器存储到内存
            }
            for (; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];  // 该行结尾处有几个元素还未计算
            }
            m[i][k] = 0;
        }
    }
}

//// Pthread 动态线程
//typedef struct {
//    int k; //消去的轮次
//    int t_id; //线程id
//}threadParam_t;
//
//// 线程函数
//DWORD WINAPI threadFunc(LPVOID param) {
//    threadParam_t* p = reinterpret_cast<threadParam_t*>(param);
//    int k = p->k;        // 消去的轮次
//    int t_id = p->t_id;  // 线程编号
//    int i = k + t_id + 1;  // 获取自己的计算任务
//
//    for (int j = k + 1; j < N; ++j) {
//        m[i][j] -= m[i][k] * m[k][j];
//    }
//    m[i][k] = 0;
//
//    return 0;
//}
//
//void pthread_LU() {
//    for (int k = 0; k < N; ++k) {
//        // 主线程做除法操作
//        for (int j = k + 1; j < N; ++j) {
//            m[k][j] = m[k][j] / m[k][k];
//        }
//        m[k][k] = 1.0;
//
//        // 创建工作线程，进行消去操作
//        int worker_count = N - 1 - k; // 工作线程数量
//        vector<thread> threads(worker_count);
//        vector<threadParam_t> params(worker_count);
//
//        // 分配任务
//        for (int t_id = 0; t_id < worker_count; ++t_id) {
//            params[t_id].k = k;
//            params[t_id].t_id = t_id;
//        }
//
//        // 创建线程
//        for (int t_id = 0; t_id < worker_count; ++t_id) {
//            threads[t_id] = thread(threadFunc, &params[t_id]);
//        }
//
//        // 主线程挂起等待所有的工作线程完成此轮消去工作
//        for (int t_id = 0; t_id < worker_count; ++t_id) {
//            threads[t_id].join();
//        }
//    }
//
//}

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

int main() {
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
    // 释放内存
    freeUnalign();


    //------------------------------------------------------------------------------
    // 优化高斯消去法部分
    //------------------------------------------------------------------------------
    // 线程优化
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
    cout << "线程优化高斯消去法时间: " << elapsedTime << " ms." << std::endl;

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


    // 释放内存
    freeUnalign();
    return 0;
}
