
#include "yolov5_thread_pool.h"
#include "draw/cv_draw.h"
// 构造函数
Yolov5ThreadPool::Yolov5ThreadPool() { stop = false; }
// 这个stop就和老杜之前写的控制线程开始和停止的变量类似的：worker_running_ = false;就这个变量
// 析构函数
Yolov5ThreadPool::~Yolov5ThreadPool()
{
    // stop all threads
    stop = true;
    cv_task.notify_all();  // 唤醒等待这条件变量的所有线程
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}
// 初始化：加载模型，创建线程，参数：模型路径，线程数量
nn_error_e Yolov5ThreadPool::setUp(std::string &model_path, int num_threads)
{
    // 遍历线程数量，创建模型实例，放入vector
    // 这些线程加载的模型是同一个
    for (size_t i = 0; i < num_threads; ++i)
    {
        std::shared_ptr<Yolov5> yolov5 = std::make_shared<Yolov5>();
        yolov5->LoadModel(model_path.c_str());
        yolov5_instances.push_back(yolov5);  // std::vector<std::shared_ptr<Yolov5>> yolov5_instances;
    }
    // 遍历线程数量，创建线程
    for (size_t i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(&Yolov5ThreadPool::worker, this, i);  // std::vector<std::thread> threads; i是id，maybe线程号
        // 回顾下trtpro里面的线程初始化：worker就是实际的工作的，初始化线程需要&这个函数，传入this
        // 见：tensorrt_pro/多线程学习demo合集/学习深度学习中涉及的线程知识.md
        // 我们来看下类成员函数作为线程启动函数的使用
        // 直接编译，thread(worker) 会提示：error: invalid use of non-static member function ‘worker()’
        // 就是说这玩意是非静态成员函数，写成静态成员函数后，就无法访问类的指针,但你可以传
        // static void worker(Yolov5ThreadPool* self)
        // threads.emplace_back(Yolov5ThreadPool::worker, this, i);
        // 上面这么写复杂了，所以才有这种写法&Yolov5ThreadPool::worker
        // 总结：在类构造函数中创建线程时，我们使用的是类成员函数的指针，而不是直接调用成员函数
        /*class Infer{
        public:

            Infer(){
                // worker_thread_ = thread(infer_worker, this);
                worker_thread_ = thread(&Infer::infer_worker, this);
            }

        private:
            // static void infer_worker(Infer* self){
            // }
            void infer_worker(){

            }

        private:
            thread worker_thread_;
        }*/
    }
    return NN_SUCCESS;
}

// 线程函数。参数：线程id
void Yolov5ThreadPool::worker(int id)
{
    while (!stop)
    {
        std::pair<int, cv::Mat> task;                            // 线程id，输入的img，不是线程ID，是frame id，我debug了，这边确实是线程ID
        std::shared_ptr<Yolov5> instance = yolov5_instances[id]; // 获取模型实例
        {
            // 获取任务
            std::unique_lock<std::mutex> lock(mtx1);
            // 因为下面的wait，这边得是唯一锁，等待的时候没必要锁住，锁住了别人就消费不了了
            cv_task.wait(lock, [&]
                         { return !tasks.empty() || stop; });

            if (stop)
            {
                return;
            }

            task = tasks.front();  // 获取先进入队列的，然后把先进入队列的弹出
            tasks.pop();
        }
        // 运行模型
        std::vector<Detection> detections;
        instance->Run(task.second, detections);  // yolov5的run
        // task.first就是frame id，一直在变多
        {
            // 保存结果
            std::lock_guard<std::mutex> lock(mtx2);         // 这把锁是当前的和cv_result的
            results.insert({task.first, detections});       // std::map 插入元素是insert，访问是find
            DrawDetections(task.second, detections);
            img_results.insert({task.first, task.second});
            cv_result.notify_one();
        }
    }
}
// 提交任务，参数：图片，id（帧号）
nn_error_e Yolov5ThreadPool::submitTask(const cv::Mat &img, int id)
{
    // 这个函数是在读取流的地方调用的，来一个图，调用一次，传入的id是当前视频多少帧，这个读流也是一个线程，但是里面有个全局变量g_pool，调用的submitTask，有点乱
    // 如果任务队列中的任务数量大于10，等待，避免内存占用过多，让给消费者不断消费，那此时就不提交了，卡在这，就是不往队列里面push{id, img}这个
    while (tasks.size() > 10)
    {
        // sleep 1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
        // 保存任务
        std::lock_guard<std::mutex> lock(mtx1);
        tasks.push({id, img});  // 一般push的时候就得加lock，这个是Yolov5ThreadPool类的tasks
    }
    cv_task.notify_one();  // 告诉Yolov5ThreadPool的worker里面，可以检查了
    return NN_SUCCESS;
}

// 获取结果，参数：检测框，id（帧号）
nn_error_e Yolov5ThreadPool::getTargetResult(std::vector<Detection> &objects, int id)
{
    // 如果没有结果，等待
    while (results.find(id) == results.end())
    {
        // sleep 1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::lock_guard<std::mutex> lock(mtx2);
    objects = results[id];
    // remove from map
    results.erase(id);

    return NN_SUCCESS;
}

// 获取结果（图片），参数：图片，id（帧号）
nn_error_e Yolov5ThreadPool::getTargetImgResult(cv::Mat &img, int id)
{
    int loop_cnt = 0;
    // 如果没有结果，等待
    while (img_results.find(id) == img_results.end())
    {
        // 等待 5ms x 1000 = 5s
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        loop_cnt++;
        if (loop_cnt > 1000)
        {
            NN_LOG_ERROR("getTargetImgResult timeout");
            return NN_TIMEOUT;
        }
    }
    std::lock_guard<std::mutex> lock(mtx2);
    img = img_results[id];
    // remove from map
    img_results.erase(id);

    return NN_SUCCESS;
}
// 停止所有线程
void Yolov5ThreadPool::stopAll()
{
    stop = true;
    cv_task.notify_all();
}