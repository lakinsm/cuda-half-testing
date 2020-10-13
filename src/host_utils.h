#ifndef NOCTURNAL_LLAMA_HOST_UTILS_H
#define NOCTURNAL_LLAMA_HOST_UTILS_H

#include <ncurses.h>
#include <functional>
#include <string>
#include <sys/time.h>
#include <vector>
#include <mutex>


class HostUtils {
public:
    HostUtils();
	HostUtils(const int &job_numbers);
	~HostUtils();

	bool fileExists(const std::string &file);
	void recordStartTime();
	void recordStopTime();
	double timeDifference();
	void initProgressBar();
	void resetProgressBar(const int &job_numbers);
	void incrementProgressBar();
private:
	int64_t _start_time;
	int64_t _stop_time;
	int _current;
	int _total;
	std::mutex _mtx;
    const int _bar_width = 70;
};

#endif // NOCTURNAL_LLAMA_HOST_UTILS_H