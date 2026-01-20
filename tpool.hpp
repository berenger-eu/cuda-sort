#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP


#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(std::size_t thread_count = std::thread::hardware_concurrency())
        : stop_(false), outstanding_(0)
    {
        if (thread_count == 0) thread_count = 1;
        workers_.reserve(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        shutdown();
    }

    // Submit non-group task; returns a future for the result.
    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
    {
        using R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;

        // Bind callable + args into a nullary function.
        auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

        std::packaged_task<R()> task(std::move(bound));
        std::future<R> fut = task.get_future();

        // Wrap packaged_task so pool can decrement outstanding on completion.
        auto job = [this, t = std::move(task)]() mutable {
            // packaged_task stores exceptions into the future automatically.
            t();
            finish_one_();
        };

        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (stop_) throw std::runtime_error("ThreadPool::submit on stopped pool");
            tasks_.push_back(TaskItem{nullptr, std::move(job)});
            outstanding_.fetch_add(1, std::memory_order_relaxed);
        }
        cv_.notify_one();

        return fut;
    }

    // Wait until pool has no queued or running tasks (group or non-group).
    void wait_for_idle() {
        std::unique_lock<std::mutex> lock(idle_mtx_);
        idle_cv_.wait(lock, [this] {
            return outstanding_.load(std::memory_order_acquire) == 0;
        });
    }

    // Stop accepting new tasks; join workers. (Does not drain queued work.)
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (stop_) return;
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
    }

    class TaskGroup {
    public:
        explicit TaskGroup(ThreadPool& pool) : pool_(pool), count_(0) {}

        TaskGroup(const TaskGroup&) = delete;
        TaskGroup& operator=(const TaskGroup&) = delete;

        // Fire-and-forget into this group.
        template <class F, class... Args>
        void run(F&& f, Args&&... args) {
            count_.fetch_add(1, std::memory_order_relaxed);

            auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

            auto group_job = [this, fn = std::move(bound)]() mutable {
                fn();

                if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    // count reached zero
                    // Wake any threads waiting on this group.
                    std::lock_guard<std::mutex> lk(m_);
                    cv_.notify_all();
                }
            };

            pool_.enqueue_group_task_(this, std::move(group_job));
        }

        // Help-while-waiting:
        // - If there are tasks of this group in the pool, run them on the calling thread.
        // - Otherwise wait until something changes.
        void wait() {
            for (;;) {
                if (count_.load(std::memory_order_acquire) == 0) return;

                // Make progress by executing one task from THIS group if possible.
                if (pool_.try_run_one_for_group_(this)) {
                    continue;
                }

                // No group task available right now. Sleep until:
                // - group finishes, OR
                // - a task for this group appears, OR
                // - pool stops.
                std::unique_lock<std::mutex> lock(pool_.mtx_);
                pool_.cv_.wait(lock, [this] {
                    return count_.load(std::memory_order_acquire) == 0
                        || pool_.stop_
                        || pool_.has_task_for_group_locked_(this);
                });
                // loop again
            }
        }

    private:
        ThreadPool& pool_;
        std::atomic<std::size_t> count_;
        std::mutex m_;
        std::condition_variable cv_;
    };

    TaskGroup make_group() { return TaskGroup(*this); }

private:
    struct TaskItem {
        TaskGroup* group;               // nullptr for non-group tasks
        std::function<void()> job;      // already wrapped to call finish_one_()
    };

    void worker_loop() {
        for (;;) {
            TaskItem item;

            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                if (stop_ && tasks_.empty()) return;

                item = std::move(tasks_.front());
                tasks_.pop_front();
            }

            item.job();
        }
    }

    void enqueue_group_task_(TaskGroup* g, std::function<void()> group_job) {
        // Wrap so pool decrements outstanding afterwards no matter what.
        auto job = [this, gj = std::move(group_job)]() mutable {
            gj();
            finish_one_();
        };

        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (stop_) throw std::runtime_error("TaskGroup::run on stopped pool");
            tasks_.push_back(TaskItem{g, std::move(job)});
            outstanding_.fetch_add(1, std::memory_order_relaxed);
        }
        cv_.notify_one();
    }

    // Must be called with mtx_ held.
    bool has_task_for_group_locked_(TaskGroup* g) const {
        for (const auto& it : tasks_) {
            if (it.group == g) return true;
        }
        return false;
    }

    // Try to find and run one queued task belonging to group g.
    // Returns true if executed one.
    bool try_run_one_for_group_(TaskGroup* g) {
        TaskItem item;
        bool found = false;

        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (std::size_t i = 0; i < tasks_.size(); ++i) {
                if (tasks_[i].group == g) {
                    item = std::move(tasks_[i]);
                    tasks_.erase(tasks_.begin() + static_cast<std::ptrdiff_t>(i));
                    found = true;
                    break;
                }
            }
        }

        if (!found) return false;
        item.job();
        return true;
    }

    void finish_one_() {
        // If we just brought outstanding to zero, notify idle waiters.
        if (outstanding_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(idle_mtx_);
            idle_cv_.notify_all();
        }
        // Wake waiters that are waiting for group tasks to appear.
        cv_.notify_all();
    }

private:
    std::vector<std::thread> workers_;

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<TaskItem> tasks_;
    bool stop_;

    std::atomic<std::size_t> outstanding_;

    std::mutex idle_mtx_;
    std::condition_variable idle_cv_;
};

#endif // THREADPOOL_HPP