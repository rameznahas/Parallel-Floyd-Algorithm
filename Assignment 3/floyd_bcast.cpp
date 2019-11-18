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

void print_adj_mat(int** mat, int n, const char* msg) {
	std::string output = "";
	std::string s = "";
	std::string m = msg;
	size_t max = m.size();

	for (int r = 0; r < n; ++r) {
		for (int c = 0; c < n; ++c) {
			if (mat[r][c] == INF) s.append("INF\t");
			else {
				s.append(std::to_string(mat[r][c]));
				s.append("\t");
			}
		}

		size_t count = s.size() + TAB * n;
		max = count > max ? count : max;
		s.append("\n");
		output.append(s);
		s = "";
	}

	print_separation(max);
	std::cout << msg << std::endl;
	print_separation(max);
	std::cout << output;
	print_separation(max);
}

void gen_directed_graph(int**& G, int n) {
	G = new int*[n];
	for (int i = 0; i < n; ++i)
		G[i] = new int[n];

	for (int r = 0; r < n; ++r)
		for (int c = 0; c < n; ++c)
			G[r][c] = -1;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(1, 100);

	for (int r = 0; r < n; ++r)
		for (int c = 0; c < n; ++c) {
			if (G[r][c] != INF) {
				int weight = dist(gen);
				weight = weight < 51 ? INF : (weight % 20) + 1;

				G[r][c] = r == c ? 0 : weight;
				if (weight != INF && r != c)
					G[c][r] = INF; // transpose idx
			}
		}
}

void cpy_directed_graph(int** G, int**& G_cpy, int n) {
	G_cpy = new int*[n];
	for (int i = 0; i < n; ++i)
		G_cpy[i] = new int[n];

	for (int r = 0; r < n; ++r)
		for (int c = 0; c < n; ++c)
			G_cpy[r][c] = G[r][c];
}

void communication(int*& row_buff, int*& col_buff, int**& g, int p_row, int p_col, int k, int dim) {
	MPI_Comm row_comm, col_comm;

	int row_key = p_col == (k / dim) ? 0 : 1;
	int col_key = p_row == (k / dim) ? 0 : 1;

	MPI_Comm_split(MPI_COMM_WORLD, p_row, row_key, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, p_col, col_key, &col_comm);

	if (row_key == 0) {
		int c = k % dim;
		for (int i = 0; i < dim; ++i)
			row_buff[i] = g[i][c];
	}

	if (col_key == 0) {
		int r = k % dim;
		for (int i = 0; i < dim; ++i)
			col_buff[i] = g[r][i];
	}
	MPI_Bcast(row_buff, dim, MPI_INT, 0, row_comm);
	MPI_Bcast(col_buff, dim, MPI_INT, 0, col_comm);
}

void serial_floyds(int**& G, int n) {
	for (int k = 0; k < n; ++k)
		for (int r = 0; r < n; ++r)
			for (int c = 0; c < n; ++c) {
				int cur_cost = G[r][c];
				int cost = G[r][k] + G[k][c];
				G[r][c] = cur_cost < cost ? cur_cost : cost;
			}
}

void parallel_floyds_bcast(int**& g, int p_row, int p_col, int n, int dim) {
	for (int k = 0; k < n; ++k) {
		int* row_buff = new int[dim];
		int* col_buff = new int[dim];

		communication(row_buff, col_buff, g, p_row, p_col, k, dim);

		for (int r = 0; r < dim; ++r)
			for (int c = 0; c < dim; ++c) {
				int cur_cost = g[r][c];
				int cost = row_buff[r] + col_buff[c];
				g[r][c] = cur_cost < cost ? cur_cost : cost;
			}

		delete[] row_buff, col_buff;
	}
}

void prep_for_partition(int*& G_buff, int** G, int p, int p_grid_width, int dim) {
	int idx = 0;
	for (int i = 0; i < p; ++i) {
		int p_r = i / p_grid_width;
		int p_c = i % p_grid_width;

		for (int r = 0; r < dim; ++r) {
			int elem_r = p_r * dim + r;
			for (int c = 0; c < dim; ++c) {
				int elem_c = p_c * dim + c;
				G_buff[idx++] = G[elem_r][elem_c];
			}
		}
	}
}

void init(int argc, char** argv, int**& G, int**& G_cpy, int**& g, int& p, int& pid, int& n, int& total_count, int& count, int& dim, int& p_grid_width, std::pair<int, int>& p_2d_idx) {
	n = argc > 1 ? std::stoi(argv[1]) : NODES; // n: num of nodes

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	total_count = n * n;
	count = total_count / p;
	float fdim = sqrtf(count);
	dim = (int)fdim;

	// if true, not a square 2D distribution. exit program
	if (dim != fdim) {
		if (pid == 0) {
			std::cout << std::endl << "Number of processes do not lead to a square 2D distribution." << std::endl;
		}
		MPI_Finalize();
		std::exit(1);
	}

	g = new int*[dim];
	for (int i = 0; i < dim; ++i) {
		g[i] = new int[dim];
	}

	// 1D buffer of size total_count = n * n, used for scattering data from G to local matrix g.
	int* G_buff = new int[total_count];

	// 1D buffer of size count = dim * dim, used for scattering data from G to local matrix g.
	int* g_buff = new int[count];

	p_grid_width = n / dim;
	p_2d_idx = { pid / p_grid_width, pid % p_grid_width };

	if (pid == 0) {
		gen_directed_graph(G, n);
		cpy_directed_graph(G, G_cpy, n);
		print_adj_mat(G, n, "cost matrix before floyd's algorithm");
		prep_for_partition(G_buff, G, p, p_grid_width, dim);

		MPI_Scatter(G_buff, count, MPI_INT, g_buff, count, MPI_INT, 0, MPI_COMM_WORLD);
		delete[] G_buff;
	}
	else {
		MPI_Scatter(G_buff, count, MPI_INT, g_buff, count, MPI_INT, 0, MPI_COMM_WORLD);
	}

	for (int i = 0; i < count; ++i) {
		int r = i / dim;
		int c = i % dim;
		g[r][c] = g_buff[i];
	}

	delete[] g_buff;
}

int main(int argc, char** argv) {
	int p, pid, n, total_count, count, dim, p_grid_width;
	int** G = nullptr, **G_cpy = nullptr, **g = nullptr;
	int* G_buff = nullptr, *g_buff = nullptr;
	std::pair<int, int> p_2d_idx;

	init(argc, argv, G, G_cpy, g, p, pid, n, total_count, count, dim, p_grid_width, p_2d_idx);

	clock_t start = clock(), Tp;
	parallel_floyds_bcast(g, p_2d_idx.first, p_2d_idx.second, n, count);
	MPI_Barrier(MPI_COMM_WORLD);

	if (pid == 0) {
		clock_t end = clock();
		Tp = (end - start) * 1000 / CLOCKS_PER_SEC;
	}

	for (int i = 0; i < count; ++i) {
		int r = i / dim;
		int c = i % dim;
		g_buff[i] = g[r][c];
	}

	if (pid == 0) {
		G_buff = new int[total_count];
		MPI_Gather(g_buff, count, MPI_INT, G_buff, count, MPI_INT, 0, MPI_COMM_WORLD);

		int idx = 0;
		for (int i = 0; i < p; ++i) {
			int p_r = i / p_grid_width;
			int p_c = i % p_grid_width;

			for (int r = 0; r < dim; ++r) {
				int elem_r = p_r * dim + r;
				for (int c = 0; c < dim; ++c) {
					int elem_c = p_c * dim + c;
					G[elem_r][elem_c] = G_buff[idx];
				}
			}
		}
	}
	else {
		MPI_Gather(g_buff, count, MPI_INT, G_buff, count, MPI_INT, 0, MPI_COMM_WORLD);
	}

	if (pid == 0) {
		start = clock();
		serial_floyds(G_cpy, n);
		clock_t end = clock();
		clock_t Ts = (end - start) * 1000 / CLOCKS_PER_SEC;

		print_adj_mat(G_cpy, n, "cost matrix after serial floyd's algorithm");
		print_adj_mat(G, n, "cost matrix after parallel floyd's algorithm");
		std::cout << "Ts: " << Ts << "ms" << std::endl;
		std::cout << "Tp: " << Tp << "ms" << std::endl;
		delete[] G, G_cpy, G_buff;
	}

	delete[] g, g_buff;

	MPI_Finalize();
}