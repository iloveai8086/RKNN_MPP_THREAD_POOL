
#include <thread>

#include <opencv2/opencv.hpp>

#include "task/yolov5.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "media/media_buffer.h"
#include "media/zlmedia_worker.h"

extern void *mpi_enc_test(int width, int height);

extern void get_source_shape(int *width, int *height);
extern int init_ffmpeg_source(const char *filepath);

int main(int argc, char **argv)
{

    const char *model_file = argv[1];                                // 模型文件路径
    std::string push_url = "rtmp://192.168.1.243:1935/live/camera1"; // 推流地址
    std::string source_url = argv[2];                                // 源视频流地址

    Yolov5 yolo;                // 实例化yolov5
    yolo.LoadModel(model_file); // 加载模型文件
    init_media_buffer();        // 初始化媒体缓冲区

    int width, height;
    std::thread t(init_ffmpeg_source, source_url.c_str()); // 开启线程初始化ffmpeg源, 读取视频帧
    get_source_shape(&width, &height);                     // 获取源视频的宽高

    std::thread t1(mpi_enc_test, width, height); // 开启线程初始化ffmpeg编码器, 编码视频帧
    init_zlmediakit(width, height, push_url);    // 初始化zlmediakit, 推流

    // 记录开始时间
    auto start_all = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    int fps = 0;
    while (true)
    {
        cv::Mat img = pop_src_media();  // 从源视频缓冲区中取出一帧视频帧
        std::vector<Detection> objects; // 存储检测结果
        yolo.Run(img, objects);         // 运行yolov5模型, 检测结果存储在objects中

        DrawDetections(img, objects); // 在视频帧上绘制检测结果

        // 算法2：计算超过 1s 一共处理了多少张图片
        frame_count++;
        // all end
        auto end_all = std::chrono::high_resolution_clock::now();
        auto elapsed_all_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count() / 1000.f;
        // 每隔1秒打印一次
        if (elapsed_all_2 > 1000)
        {
            fps = frame_count / (elapsed_all_2 / 1000.0f);
            NN_LOG_INFO("Method2 Time:%fms, FPS:%f, Frame Count:%d", elapsed_all_2, fps, frame_count);
            start_all = std::chrono::high_resolution_clock::now();
            frame_count = 0;
        }
        // 绘制总耗时和帧率
        cv::putText(img, std::to_string(fps) + " fps", cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

        push_out_media(img); // 将绘制了检测结果的视频帧推送到编码器缓冲区
    }

    return 0;
}