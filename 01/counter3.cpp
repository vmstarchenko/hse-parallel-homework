#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

void sleep(unsigned milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

class MeanCounterBase {
public:
    virtual ~MeanCounterBase() {
    }

    virtual double mean() = 0;
    virtual void add(int value) = 0;

    bool check(double expectedMean) const {
        return std::abs(expectedMean - calcMean()) <
               std::numeric_limits<double>::epsilon();
    }

protected:
    double calcMean() const {
        sleep(1000);
        return (double)sum / count;
    }

    void doAdd(int value) {
        sleep(1000);
        sum += value;
        sleep(1000);
        count += 1;
    }

private:
    long sum = 0;
    long count = 0;
};

void writer(int id, MeanCounterBase& counter, int value) {
    std::printf("[%d] Writer started\n", id);
    counter.add(value);
    std::printf("[%d] Added value: %d\n", id, value);
}

void reader(int id, MeanCounterBase& counter) {
    std::printf("[%d] Reader started\n", id);
    double mean = counter.mean();
    std::printf("[%d] Read mean: %f\n", id, mean);
}

bool check(MeanCounterBase& counter) {
    return counter.check(counter.mean());
}

// === DO NOT REMOVE THIS LINE ===
//// Your solution below

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

class MeanCounter : public MeanCounterBase {
public:
    double mean() override {
        std::unique_lock<std::mutex> ulock(mutex);

        // start read
        active_readers++;
        while (position == 0) { // has active writer
            read_cond_var.wait(ulock);
        }
        position = 1; // activate reader
        ulock.unlock();

        auto res = calcMean();

        // finish read
        ulock.lock();
        active_readers--;
        if (active_readers == 0) {
            position = 2; // deactivate reader
            write_cond_var.notify_one();
        }

        return res;
    }

    void add(int value) override {
        std::unique_lock<std::mutex> ulock(mutex);

        // start write
        while (position != 2) { // has active readers o active writer
            write_cond_var.wait(ulock);
        }
        position = 0; // activate writer
        ulock.unlock();

        doAdd(value);

        // finish write
        ulock.lock();
        position = 2; // deactivate writer
        if (active_readers) {
            read_cond_var.notify_all();
        } else {
            write_cond_var.notify_one();
        }
        ulock.unlock();
    }

private:
    std::mutex mutex;

    std::condition_variable read_cond_var;
    std::condition_variable write_cond_var;

    std::atomic<int> active_readers{0};
    std::atomic<int> position{2};
    std::atomic<int> inactive_writer{0};
};

//// Your solution above
// === DO NOT REMOVE THIS LINE ===

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %position trace_file\n", argv[0]);
        return 1;
    }

    MeanCounter counter;
    std::vector<std::thread> threads;

    std::ifstream f(argv[1]);
    int thread_id = 0;
    while (f) {
        std::string op;
        f >> op;
        if (op == "W") {
            int value;
            f >> value;
            threads.emplace_back(writer, thread_id, std::ref(counter), value);
        } else if (op == "R") {
            threads.emplace_back(reader, thread_id, std::ref(counter));
        } else if (!op.empty()) {
            std::fprintf(stderr, "Unknown op: %position\n", op.c_str());
            break;
        }
        thread_id++;
        sleep(500);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (!check(counter)) {
        std::fprintf(stderr, "Make sure you are using MeanCounterBase\n");
        return 2;
    }
}
