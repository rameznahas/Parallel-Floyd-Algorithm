#include <mpi.h>
#include <time.h>
#include <math.h>
#include <random>
#include <iostream>
#include <string>
#define INF 100000
#define NODES 8
#define TAB 5

void malloc_2d_matrix(int**& mat, int n) {
	mat = new int*[n];
	for (int i = 0; i < n; ++i)
		mat[i] = new int[n];
}

void free_2d_matrix(int**& mat, int n) {
	for (int i = 0; i < n; ++i)
		delete[] mat[i];
	delete[] mat;
}

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

/*
	Generates a random directed graph and saves it as adjacency matrix G.
*/
void gen_directed_graph(int**& G, int n) {
	malloc_2d_matrix(G, n);

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

void serial_floyds(int**& G, int n) {
	for (int k = 0; k < n; ++k)
		for (int r = 0; r < n; ++r)
			for (int c = 0; c < n; ++c) {
				int cur_cost = G[r][c];
				int cost = G[r][k] + G[k][c];
				G[r][c] = cur_cost < cost ? cur_cost : cost;
			}
}

/*
	Communication step that is done during parallel floyd's algorithm.
*/
void communication(int*& row_buff, int*& col_buff, int**& g, int p_row, int p_col, int k, int dim) {
	MPI_Comm row_comm, col_comm;

	int row_key = p_col == (k / dim) ? 0 : 1; // determines which process is the root in the row_comm
	int col_key = p_row == (k / dim) ? 0 : 1; // determines which process is the root in the col_comm

	MPI_Comm_split(MPI_COMM_WORLD, p_row, row_key, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, p_col, col_key, &col_comm);

	// if true, is root.
	// so copy elements that will be bcasted along the row_comm to row_buff
	if (row_key == 0) {
		int c = k % dim;
		for (int i = 0; i < dim; ++i)
			row_buff[i] = g[i][c];
	}

	// if true, is root.
	// so copy elements that will be bcasted along the col_comm to col_buff
	if (col_key == 0) {
		int r = k % dim;
		for (int i = 0; i < dim; ++i)
			col_buff[i] = g[r][i];
	}
	MPI_Bcast(row_buff, dim, MPI_INT, 0, row_comm);
	MPI_Bcast(col_buff, dim, MPI_INT, 0, col_comm);
}

/*
	Parallel formulation of floyd's algorithm.
*/
void parallel_floyds_bcast(int**& g, int p_row, int p_col, int n, int dim) {
	for (int k = 0; k < n; ++k) {
		int* row_buff = new int[dim];
		int* col_buff = new int[dim];

		communication(row_buff, col_buff, g, p_row, p_col, k, dim);

		// local computation
		for (int r = 0; r < dim; ++r)
			for (int c = 0; c < dim; ++c) {
				int cur_cost = g[r][c];
				int cost = row_buff[r] + col_buff[c];
				g[r][c] = cur_cost < cost ? cur_cost : cost;
			}

		delete[] row_buff, col_buff;
	}
}

/*
	Flattens 2D buffer G into a 1D buffer.

	Flattening is not done row by row, but 2D-block by 2D-block.
	This is done to respect the 2D partition order of G before scattering it.
*/
void flatten_for_scatter(int*& G_buff, int** G, int p, int p_grid_width, int dim) {
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

/*
	Unflattens 1D buffer G_buff back to its initial 2D configuration.
*/
void unflatten_after_gather(int**& G, int* G_buff, int p, int p_grid_width, int dim) {
	int idx = 0;
	for (int i = 0; i < p; ++i) {
		int p_r = i / p_grid_width;
		int p_c = i % p_grid_width;

		for (int r = 0; r < dim; ++r) {
			int elem_r = p_r * dim + r;
			for (int c = 0; c < dim; ++c) {
				int elem_c = p_c * dim + c;
				G[elem_r][elem_c] = G_buff[idx++];
			}
		}
	}
}

/*
	Initializes the program.

		- Initializes MPI and all the necessary program variables.
		- Allocates heap memory for the necessary memory buffers.
		- Makes sure the 2D distribution is respected and aborts if it isn't.
		- Generates random directed graph in process 0 and scatters it to all processes.
*/
void init(int argc, char** argv, int**& G, int*& G_buff, int**& g, int*& g_buff, int& p, int& pid, int& n, int& count, int& dim, int& p_grid_width, std::pair<int, int>& p_2d_idx) {
	n = argc > 1 ? std::stoi(argv[1]) : NODES; // n: num of nodes

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);

	int total_count = n * n;
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

	malloc_2d_matrix(g, dim);

	g_buff = new int[count];

	p_grid_width = n / dim;
	p_2d_idx = { pid / p_grid_width, pid % p_grid_width }; // 2D id of process

	if (pid == 0) {
		G_buff = new int[total_count];

		gen_directed_graph(G, n);
		print_adj_mat(G, n, "cost matrix before floyd's algorithm");
		flatten_for_scatter(G_buff, G, p, p_grid_width, dim);

		MPI_Scatter(G_buff, count, MPI_INT, g_buff, count, MPI_INT, 0, MPI_COMM_WORLD);
	}
	else {
		MPI_Scatter(G_buff, count, MPI_INT, g_buff, count, MPI_INT, 0, MPI_COMM_WORLD);
	}

	for (int i = 0; i < count; ++i) {
		int r = i / dim;
		int c = i % dim;
		g[r][c] = g_buff[i];
	}
}

int main(int argc, char** argv) {
	int p, pid, n, count, dim, p_grid_width;
	int** G = nullptr, **g = nullptr;
	int* G_buff = nullptr, *g_buff = nullptr;
	std::pair<int, int> p_2d_idx;
	clock_t start, end, Tp, Ts;

	init(argc, argv, G, G_buff, g, g_buff, p, pid, n, count, dim, p_grid_width, p_2d_idx);

	if (pid == 0) start = clock();
	parallel_floyds_bcast(g, p_2d_idx.first, p_2d_idx.second, n, dim);

	MPI_Barrier(MPI_COMM_WORLD);

	// parallel floyd done. get Tp, now do serial floyds and get Ts.
	if (pid == 0) {
		end = clock();
		Tp = (end - start) * 1000 / CLOCKS_PER_SEC;

		start = clock();
		serial_floyds(G, n);
		end = clock();
		Ts = (end - start) * 1000 / CLOCKS_PER_SEC;

		print_adj_mat(G, n, "cost matrix after serial floyd's algorithm");
	}

	// flatten local matrix before gathering at root
	for (int i = 0; i < count; ++i) {
		int r = i / dim;
		int c = i % dim;
		g_buff[i] = g[r][c];
	}

	if (pid == 0) {
		MPI_Gather(g_buff, count, MPI_INT, G_buff, count, MPI_INT, 0, MPI_COMM_WORLD);
		unflatten_after_gather(G, G_buff, p, p_grid_width, dim);
		
		print_adj_mat(G, n, "cost matrix after parallel floyd's algorithm");
		std::cout << "Ts: " << Ts << "ms" << std::endl;
		std::cout << "Tp: " << Tp << "ms" << std::endl;

		// cleanup
		free_2d_matrix(G, n);
		delete[] G_buff;
	}
	else {
		MPI_Gather(g_buff, count, MPI_INT, G_buff, count, MPI_INT, 0, MPI_COMM_WORLD);
	}

	// cleanup
	free_2d_matrix(g, dim);
	delete[] g_buff;

	MPI_Finalize();
}