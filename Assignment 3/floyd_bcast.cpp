#include <mpi.h>
#include <time.h>
#include <math.h>
#include <random>
#include <iostream>
#include <string>
#define INF 100000
#define NODES 8
#define TAB 5

void print_separation(int count) {
	std::string s = "";
	for (int i = 0; i < count; ++i)
		s += "=";
	std::cout << s << std::endl;
}

void print_adj_mat(const int* mat, int n, const char* msg) {
	int size = n * n;

	std::string output = "";
	std::string s = "";
	std::string m = msg;
	size_t max = m.size();

	for (int i = 0; i < size; ++i) {
		int row = i / n;
		int col = i % n;

		if (mat[i] == INF) s.append("INF\t");
		else {
			s.append(std::to_string(mat[i]));
			s.append("\t");
		}

		if (col == n - 1) {
			size_t count = s.size() + TAB * n;
			max = count > max ? count : max;
			s.append("\n");
			output.append(s);
			s = "";
		} 
	}

	print_separation(max);
	std::cout << msg << std::endl;
	print_separation(max);
	std::cout << output;
	print_separation(max);
}

void gen_directed_graph(int*& G, int n) {
	int total_count = n * n;
	G = new int[total_count];

	for (int i = 0; i < total_count; ++i) {
		G[i] = -1;
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(1, 100);

	for (int i = 0; i < total_count; ++i) {
		if (G[i] != INF) {
			int weight = dist(gen);
			weight = weight < 51 ? INF : (weight % 20) + 1;

			int row = i / n;
			int col = i % n;
			int transpose_idx = col * n + row;

			G[i] = row == col ? 0 : weight;
			if (weight != INF && row != col)
				G[transpose_idx] = INF;
		}
	}
}

void cpy_directed_graph(const int* G, int*& G_cpy, int total_count) {
	G_cpy = new int[total_count];

	for (int i = 0; i < total_count; ++i)
		G_cpy[i] = G[i];
}

void serial_floyds(int*& G, int n) {
	for (int k = 0; k < n; ++k)
		for (int row = 0; row < n; ++row)
			for (int col = 0; col < n; ++col) {
				int idx = row * n + col;
				int idx_k_col = row * n + k;
				int idx_k_row = k * n + col;

				int cur_cost = G[idx];
				int cost = G[idx_k_col] + G[idx_k_row];
				G[idx] = cur_cost < cost ? cur_cost : cost;
			}
}

void init(int argc, char** argv, int*& G, int*& G_cpy, int*& g, int& p, int& pid, int& n, int& total_count, int& count) {
	n = argc > 1 ? std::stoi(argv[1]) : NODES; // n: num of nodes

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	total_count = n * n;
	count = total_count / p;

	// if true, not a 1D row distribution. exit program
	if (count > n || (n % count) != 0) {
		if (pid == 0) {
			std::cout << std::endl << "Number of processes do not lead to a 1D row distribution." << std::endl;
		}
		MPI_Finalize();
		std::exit(1);
	}

	g = new int[count];

	if (pid == 0) {
		gen_directed_graph(G, n);
		cpy_directed_graph(G, G_cpy, total_count);

		print_adj_mat(G, n, "cost matrix before floyd's algorithm");
		MPI_Scatter(G, count, MPI_INT, g, count, MPI_INT, 0, MPI_COMM_WORLD);
		delete[] G;
	}
	else {
		MPI_Scatter(G, count, MPI_INT, g, count, MPI_INT, 0, MPI_COMM_WORLD);
	}
}

void communication(int& row_buff, int*& col_buff, int*& g, int p_row, int p_col, int k, int count) {
	MPI_Comm row_comm, col_comm;

	int row_key = p_col == (k / count) ? 0 : 1;
	int col_key = p_row == k ? 0 : 1;

	MPI_Comm_split(MPI_COMM_WORLD, p_row, row_key, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, p_col, col_key, &col_comm);

	if (row_key == 0) {
		row_buff = g[k % count];
	}

	if (col_key == 0) {
		for (int i = 0; i < count; ++i)
			col_buff[i] = g[i];
	}
	MPI_Bcast(&row_buff, 1, MPI_INT, 0, row_comm);
	MPI_Bcast(col_buff, count, MPI_INT, 0, col_comm);
}

int main(int argc, char** argv) {
	int p, pid, n, total_count, count;
	int* G = nullptr, *G_cpy = nullptr, *g = nullptr;

	init(argc, argv, G, G_cpy, g, p, pid, n, total_count, count);
	
	int p_grid_width = n / count;
	std::pair<int, int> p_coord = { pid / p_grid_width, pid % p_grid_width };

	clock_t start = clock();

	for (int k = 0; k < n; ++k) {
		int row_buff;
		int* col_buff = new int[count];

		communication(row_buff, col_buff, g, p_coord.first, p_coord.second, k, count);
		
		for (int i = 0; i < count; ++i) {
			int cur_cost = g[i];
			int cost = row_buff + col_buff[i];
			g[i] = cur_cost < cost ? cur_cost : cost;
		}
		delete[] col_buff;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (pid == 0) {
		clock_t end = clock();
		clock_t Tp = (end - start) * 1000 / CLOCKS_PER_SEC;

		G = new int[total_count];
		MPI_Gather(g, count, MPI_INT, G, count, MPI_INT, 0, MPI_COMM_WORLD);

		start = clock();
		serial_floyds(G_cpy, n);
		end = clock();
		clock_t Ts = (end - start) * 1000 / CLOCKS_PER_SEC;

		print_adj_mat(G_cpy, n, "cost matrix after serial floyd's algorithm");
		print_adj_mat(G, n, "cost matrix after parallel floyd's algorithm");
		std::cout << "Ts: " << Ts << "ms" << std::endl;
		std::cout << "Tp: " << Tp << "ms" << std::endl;
		delete[] G, G_cpy;
	}
	else {
		MPI_Gather(g, count, MPI_INT, G, count, MPI_INT, 0, MPI_COMM_WORLD);
	}

	delete[] g;

	MPI_Finalize();
}