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

class MeanCounter : public MeanCounterBase {
public:
    double mean() override {
        m1.lock();
       // printf("start read\n");
        std::unique_lock<std::mutex> ulock(c1_m);
       // printf("kek %d\n", (int)w1);
        bool locked = true;
        while (w1) {
           // printf("wate\n");
            c1.wait(ulock);
           // printf("awake\n");
            locked = false;
        }
        if (locked) {
            ulock.unlock();
        }

        r1++;
       // printf("start start read\n");
        m1.unlock();

        auto res = calcMean();

        // m1.lock();
       // printf("finish read\n");
        r1--;
        if (r1 == 0) {
            c1.notify_all();
        }
        // m1.unlock();
       // printf("finish finish read\n");
        return res;
    }

    void add(int value) override {
        m2.lock();
       // printf("s write\n");
        std::unique_lock<std::mutex> ulock(c2_m);
        bool locked = true;
        while (w1 or r1 > 0) {
            c1.wait(ulock);
            locked = false;
        }
        if (locked) {
            ulock.unlock();
        }

        w1 = true;
       // printf("start write\n");

        doAdd(value);

        w1 = false;
        c1.notify_all();
       // printf("finish write\n");
        m2.unlock();
    }

private:
    std::mutex m1;
    std::condition_variable c1;
    std::mutex c1_m;
    std::condition_variable c2;
    std::mutex c2_m;
    std::atomic<int> r1{0}; //(number of readers waiting)
    std::atomic<int> w1{0}; // (writer waiting)

    std::mutex m2;
    int r2 = 0;  //(number of readers waiting)
    bool w2 = 0; // (writer waiting)
};

//// Your solution above
// === DO NOT REMOVE THIS LINE ===

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s trace_file\n", argv[0]);
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
            std::fprintf(stderr, "Unknown op: %s\n", op.c_str());
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
