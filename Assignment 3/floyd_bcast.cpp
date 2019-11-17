#include <mpi.h>
#include <time.h>
#include <math.h>
#include <random>
#include <iostream>
#include <string>
#define INF 100000
#define NODES 8

int main(int argc, char** argv) {
	int p, pid, n = argc > 1 ? std::stoi(argv[1]) : NODES; // n: num of nodes

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	int count = (n * n) / p;
	int p_grid_width = n / count;
	std::pair<int, int> pid_coord = { pid / p_grid_width, pid % p_grid_width };
	int* G = nullptr;
	int* g = new int[count];
	
	if (pid == 0) {
		/*G = new int[n * n]{
			0, 1, 3, 4,
			INF, 0, 1, INF,
			INF, INF, 0, 1,
			INF, INF, INF, 0
		};*/
		G = new int[n * n] {
			0, 2, 3, INF, INF, INF, INF, INF,
			INF, 0, INF, 3, 2, INF, INF, INF,
			INF, INF, 0, INF, INF, 7, INF, INF,
			INF, INF, INF, 0, 1, INF, 4, INF,
			INF, INF, INF, INF, 0, 1, INF, INF,
			INF, INF, INF, INF, INF, 0, 1, 4,
			INF, INF, INF, INF, INF, INF, 0, 3,
			INF, INF, INF, INF, INF, INF, INF, 0
		};
		MPI_Scatter(G, count, MPI_INT, g, count, MPI_INT, 0, MPI_COMM_WORLD);
		delete[] G;
	}
	else {
		MPI_Scatter(G, count, MPI_INT, g, count, MPI_INT, 0, MPI_COMM_WORLD);
	}

	clock_t start = clock();

	for (int k = 0; k < n; ++k) {
		MPI_Comm row_comm, col_comm;

		int row_key = pid_coord.second == (k / count) ? 0 : 1;
		int col_key = pid_coord.first == k ? 0 : 1;

		MPI_Comm_split(MPI_COMM_WORLD, pid_coord.first, row_key, &row_comm);
		MPI_Comm_split(MPI_COMM_WORLD, pid_coord.second, col_key, &col_comm);

		int* row_buff = new int[count];
		if (row_key == 0) {
			for (int i = 0; i < count; ++i)
				row_buff[i] = g[i];
		}
		MPI_Bcast(row_buff, count, MPI_INT, 0, row_comm);

		int* col_buff = new int[count];
		if (col_key == 0) {
			for (int i = 0; i < count; ++i)
				col_buff[i] = g[i];
		}
		MPI_Bcast(col_buff, count, MPI_INT, 0, col_comm);

		for (int i = 0; i < count; ++i) {
			int cur_cost = g[i];
			int cost = row_buff[k % count] + col_buff[i];
			g[i] = cur_cost < cost ? cur_cost : cost;
		}
		delete[] row_buff, col_buff;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (pid == 0) {
		clock_t end = clock();
		std::cout << "runtime: " << (end - start) * 1000 / CLOCKS_PER_SEC << "ms" << std::endl;

		int size = n * n;
		G = new int[size];
		MPI_Gather(g, count, MPI_INT, G, count, MPI_INT, 0, MPI_COMM_WORLD);
		std::cout << G[0] << "\t";
		for (int i = 1; i < size; ++i) {
			if (i % n == 0)
				std::cout << std::endl;
			std::cout << G[i] << "\t";
		}
		std::cout << std::endl;
		delete[] G;
	}
	else {
		MPI_Gather(g, count, MPI_INT, G, count, MPI_INT, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
}