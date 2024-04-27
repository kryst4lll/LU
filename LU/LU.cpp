#include <iostream>
#include <stdlib.h>
#include <malloc.h> // 包含 _aligned_malloc 函数
#include <immintrin.h>
#include<windows.h>
#include<xmmintrin.h>

using namespace std;

#define N 1000
#define ALIGNMENT 16 // 对齐方式为 16 字节

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

int main() {
    LARGE_INTEGER frequency;        // 声明一个LARGE_INTEGER类型的变量来存储频率
    LARGE_INTEGER start, end;       // 声明两个LARGE_INTEGER类型的变量来存储开始和结束的计数值
    double elapsedTime;             // 声明一个double类型的变量来存储经过的时间
    // 获取计时器的频率
    QueryPerformanceFrequency(&frequency);
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

    m_reset();
    // 记录开始时间
    QueryPerformanceCounter(&start);
    SIMD_LU(m);
    // 记录结束时间
    QueryPerformanceCounter(&end);
    // 计算经过的时间
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "优化高斯消去法时间: " << elapsedTime << " ms." << std::endl;
    // 释放内存
    freeUnalign();
    return 0;
}
