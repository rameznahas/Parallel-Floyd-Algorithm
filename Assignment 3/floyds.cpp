#include <mpi.h>
#include <time.h>
#include <math.h>
#include <random>
#include <iostream>
#include <string>
#define INF 100000
#define NODES 8
#define TAB 8
#define PRINT_CUTOFF 26
#define RESULT_SEPARATION 32

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

void append_separation(std::string& output, int count) {
	for (int i = 0; i < count; ++i)
		output.append("=");
	output.append("\n");
}

void print_program_results(int n, int p, int Ts, int Tp_bcast, int Tp_pipelined) {
	std::string separation = "";
	int separation_count = n < PRINT_CUTOFF ? TAB * n : RESULT_SEPARATION;

	append_separation(separation, separation_count);

	std::cout << separation;
	std::cout << "For n = " << n << " & p = " << p << std::endl << std::endl;
	std::cout << "Ts:\t\t\t" << Ts << "ms" << std::endl;
	std::cout << "Tp_bcast:\t\t" << Tp_bcast << "ms" << std::endl;
	std::cout << "Tp_pipelined:\t\t" << Tp_pipelined << "ms" << std::endl << std::endl;

	if (Tp_bcast != 0) std::cout << "SU_serial_bcast:\t" << (float)Ts / Tp_bcast << std::endl;
	else if (Tp_bcast < Ts) std::cout << "SU_serial_bcast:\tINF" << std::endl;
	else std::cout << "SU_serial_bcast:\t0" << std::endl;

	if (Tp_pipelined != 0) {
		std::cout << "SU_serial_pipelined:\t" << (float)Ts / Tp_pipelined << std::endl;
		std::cout << "SU_bcast_pipelined:\t" << (float)Tp_bcast / Tp_pipelined << std::endl;
	}
	else {
		if (Tp_pipelined < Ts) std::cout << "SU_serial_pipelined:\tINF" << std::endl;
		else std::cout << "SU_serial_pipelined:\t0" << std::endl;

		if (Tp_pipelined < Tp_bcast) std::cout << "SU_pipelined_bcast:\tINF" << std::endl;
		else std::cout << "SU_pipelined_bcast:\t0" << std::endl;
	}

	std::cout << separation;
}

void print_adj_mat(int** mat, int n, const char* msg, bool print) {
	if (!print) return;

	std::string output = "";
	int separation_count = TAB * n;
	append_separation(output, separation_count);
	output.append(msg);
	output.append("\n");
	append_separation(output, separation_count);

	for (int r = 0; r < n; ++r) {
		for (int c = 0; c < n; ++c) {
			if (mat[r][c] == INF) output.append("INF\t");
			else {
				output.append(std::to_string(mat[r][c]));
				output.append("\t");
			}
		}
		output.append("\n");
	}
	append_separation(output, separation_count);

	std::cout << output;
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
	Communication step that is done during parallel bcast floyd's algorithm.
*/
void communication_bcast(int*& row_buff, int*& col_buff, int**& g, int pid, std::pair<int, int> p_2d_idx, int k, int dim, int p_grid_width) {
	MPI_Comm row_comm, col_comm;

	int p_row = p_2d_idx.first;
	int p_col = p_2d_idx.second;

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
	Communication step that is done during parallel pipelined floyd's algorithm.
*/
void communication_pipelined(int*& row_buff, int*& col_buff, int**& g, int pid, std::pair<int, int> p_2d_idx, int k, int dim, int p_grid_width) {
	int target = k / dim;

	int p_row = p_2d_idx.first;
	int p_col = p_2d_idx.second;

	bool row_root = p_col == target ? true : false; // determines which process is the root in the row_comm
	bool col_root = p_row == target ? true : false; // determines which process is the root in the col_comm

	int upper_left_edge = 0;
	int lower_right_edge = p_grid_width - 1;

	int row_r_neighbour = pid + 1;
	int row_l_neighbour = pid - 1;

	// if true, is root.
	// so copy elements that will be sent to neighbour processes along the row_comm
	if (row_root) {
		int c = k % dim;
		for (int i = 0; i < dim; ++i)
			row_buff[i] = g[i][c];

		// if true, not at left edge, so has a left neighbour
		if (p_col != upper_left_edge)
			MPI_Send(row_buff, dim, MPI_INT, row_l_neighbour, 0, MPI_COMM_WORLD);

		// if true, not at right edge, so has a right neighbour
		if (p_col != lower_right_edge)
			MPI_Send(row_buff, dim, MPI_INT, row_r_neighbour, 0, MPI_COMM_WORLD);
	}
	else { // not root, so receive elements from root and forward them to neighbour processes along row
		// determine row root pid
		int root_pid = p_row * p_grid_width + target;

		// if true, current process is left of root process
		// so receive from right neighbour
		if (pid < root_pid) {
			MPI_Recv(row_buff, dim, MPI_INT, row_r_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// if true, not at left edge, so has a left neighbour
			if (p_col != upper_left_edge)
				MPI_Send(row_buff, dim, MPI_INT, row_l_neighbour, 0, MPI_COMM_WORLD);
		}
		else { // current process is right of root process
			MPI_Recv(row_buff, dim, MPI_INT, row_l_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// if true, not at right edge, so has a right neighbour
			if (p_col != lower_right_edge)
				MPI_Send(row_buff, dim, MPI_INT, row_r_neighbour, 0, MPI_COMM_WORLD);
		}
	}

	int col_l_neighbour = pid + p_grid_width;
	int col_u_neighbour = pid - p_grid_width;

	// if true, is root.
	// so copy elements that will be sent to neighbour processes along the col_comm
	if (col_root) {
		int r = k % dim;
		for (int i = 0; i < dim; ++i)
			col_buff[i] = g[r][i];

		// if true, not at upper edge, so has an upper neighbour
		if (p_row != upper_left_edge)
			MPI_Send(col_buff, dim, MPI_INT, col_u_neighbour, 1, MPI_COMM_WORLD);

		// if true, not at lower edge, so has a lower neighbour
		if (p_row != lower_right_edge)
			MPI_Send(col_buff, dim, MPI_INT, col_l_neighbour, 1, MPI_COMM_WORLD);
	}
	else { // not root, so receive elements from root and forward them to neighbour processes along col
		// determine col root pid
		int root_pid = target * p_grid_width + p_col;

		// if true, current process is a lower than root process in col
		// so receive from upper neighbour
		if (pid < root_pid) {
			MPI_Recv(col_buff, dim, MPI_INT, col_l_neighbour, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// if true, not at upper edge, so has an upper neighbour
			if (p_row != upper_left_edge)
				MPI_Send(col_buff, dim, MPI_INT, col_u_neighbour, 1, MPI_COMM_WORLD);
		}
		else { // current process is above root process in col
			MPI_Recv(col_buff, dim, MPI_INT, col_u_neighbour, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// if true, not at lower edge, so has a lower neighbour
			if (p_row != lower_right_edge)
				MPI_Send(col_buff, dim, MPI_INT, col_l_neighbour, 1, MPI_COMM_WORLD);
		}
	}
}

/*
	Parallel formulation of floyd's algorithm.

	@param void (*communication)(int*& row_buff, int*& col_buff, int**& g, int p_row, int p_col, int k, int dim, int p_grid_width)
		- pointer to a function that handles communication (for example bcast cpmmunication or pipelined communication)

*/
void parallel_floyds(int**& g, int*& g_buff, int pid, std::pair<int, int> p_2d_idx, int n, int dim, int p_grid_width, int count, void(*communication)(int*& row_buff, int*& col_buff, int**& g, int pid, std::pair<int, int> p_2d_idx, int k, int dim, int p_grid_width)) {
	// unflatten before computation
	for (int i = 0; i < count; ++i) {
		int r = i / dim;
		int c = i % dim;
		g[r][c] = g_buff[i];
	}

	int* row_buff = new int[dim];
	int* col_buff = new int[dim];

	for (int k = 0; k < n; ++k) {
		communication(row_buff, col_buff, g, pid, p_2d_idx, k, dim, p_grid_width);

		// local computation
		for (int r = 0; r < dim; ++r)
			for (int c = 0; c < dim; ++c) {
				int cur_cost = g[r][c];
				int cost = row_buff[r] + col_buff[c];
				g[r][c] = cur_cost < cost ? cur_cost : cost;
			}
	}

	delete[] row_buff, col_buff;
}

/*
	Initializes the program.

		- Initializes MPI and all the necessary program variables.
		- Allocates heap memory for the necessary memory buffers 
			(2 different local buffers per process: 1 for bcast floyds, 1 for pipelined floyds).
		- Makes sure the 2D distribution is respected and aborts if it isn't.
		- Generates random directed graph in process 0 and scatters it to all processes.
*/
void init(int argc, char** argv, int**& G, int*& G_buff_bcast, int*& G_buff_pipelined, int**& g, int*& g_buff_bcast, int*& g_buff_pipelined, int& p, int& pid, int& n, int& count, int& dim, int& p_grid_width, std::pair<int, int>& p_2d_idx) {
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

	g_buff_bcast = new int[count];
	g_buff_pipelined = new int[count];

	p_grid_width = n / dim;
	p_2d_idx = { pid / p_grid_width, pid % p_grid_width }; // 2D id of process

	if (pid == 0) {
		G_buff_bcast = new int[total_count];
		G_buff_pipelined = new int[total_count];

		gen_directed_graph(G, n);
		print_adj_mat(G, n, "cost matrix before floyd's algorithm", n < PRINT_CUTOFF);
		flatten_for_scatter(G_buff_bcast, G, p, p_grid_width, dim);
		flatten_for_scatter(G_buff_pipelined, G, p, p_grid_width, dim);

		MPI_Scatter(G_buff_bcast, count, MPI_INT, g_buff_bcast, count, MPI_INT, 0, MPI_COMM_WORLD); // one buffer for bcast floyd's
		MPI_Scatter(G_buff_pipelined, count, MPI_INT, g_buff_pipelined, count, MPI_INT, 0, MPI_COMM_WORLD); // one buffer for pipelined floyd's
	}
	else {
		MPI_Scatter(G_buff_bcast, count, MPI_INT, g_buff_bcast, count, MPI_INT, 0, MPI_COMM_WORLD); // one buffer for bcast floyd's
		MPI_Scatter(G_buff_pipelined, count, MPI_INT, g_buff_pipelined, count, MPI_INT, 0, MPI_COMM_WORLD); // one buffer for pipelined floyd's
	}
}

int main(int argc, char** argv) {
	int p, pid, n, count, dim, p_grid_width;
	int** G = nullptr, **g = nullptr;
	int* G_buff_bcast = nullptr, *g_buff_bcast = nullptr;
	int* G_buff_pipelined = nullptr, *g_buff_pipelined = nullptr;

	std::pair<int, int> p_2d_idx;
	clock_t start, end, Tp_bcast, Tp_pipelined, Ts;

	init(argc, argv, G, G_buff_bcast, G_buff_pipelined, g, g_buff_bcast, g_buff_pipelined, p, pid, n, count, dim, p_grid_width, p_2d_idx);

	// first do floyds serially, get Ts and print adjacency matrix of least cost paths.
	if (pid == 0) {
		start = clock();
		serial_floyds(G, n);
		end = clock();
		Ts = (end - start) * 1000 / CLOCKS_PER_SEC;
		print_adj_mat(G, n, "cost matrix after serial floyd's algorithm", n < PRINT_CUTOFF);
	}

	// wait for root process to finish serial floyd.
	MPI_Barrier(MPI_COMM_WORLD);
	
	if (pid == 0) start = clock();
	// start parallel bcast floyds
	parallel_floyds(g, g_buff_bcast, pid, p_2d_idx, n, dim, p_grid_width, count, communication_bcast);

	// wait for all processes to be done with parallel bcast floyds.
	MPI_Barrier(MPI_COMM_WORLD);

	// parallel bcast floyd done. get Tp_bcast.
	if (pid == 0) {
		end = clock();
		Tp_bcast = (end - start) * 1000 / CLOCKS_PER_SEC;
	}

	// flatten local matrix before gathering at root
	for (int i = 0; i < count; ++i) {
		int r = i / dim;
		int c = i % dim;
		g_buff_bcast[i] = g[r][c];
	}

	// gather parallel bcast floyds results at root node and print adjacency matrix.
	if (pid == 0) {
		MPI_Gather(g_buff_bcast, count, MPI_INT, G_buff_bcast, count, MPI_INT, 0, MPI_COMM_WORLD);
		unflatten_after_gather(G, G_buff_bcast, p, p_grid_width, dim);

		print_adj_mat(G, n, "cost matrix after parallel bcast floyd's algorithm", n < PRINT_CUTOFF);

		// cleanup
		delete[] G_buff_bcast;
	}
	else {
		MPI_Gather(g_buff_bcast, count, MPI_INT, G_buff_bcast, count, MPI_INT, 0, MPI_COMM_WORLD);
	}

	delete[] g_buff_bcast;

	// wait for result display.
	MPI_Barrier(MPI_COMM_WORLD);

	if (pid == 0) start = clock();
	// start parallel pipelined floyds
	parallel_floyds(g, g_buff_pipelined, pid, p_2d_idx, n, dim, p_grid_width, count, communication_pipelined);

	// wait for all processes to be done with parallel pipelined floyds.
	MPI_Barrier(MPI_COMM_WORLD);

	// parallel pipelined floyd done. get Tp_pipelined.
	if (pid == 0) {
		end = clock();
		Tp_pipelined = (end - start) * 1000 / CLOCKS_PER_SEC;
	}

	// flatten local matrix before gathering at root
	for (int i = 0; i < count; ++i) {
		int r = i / dim;
		int c = i % dim;
		g_buff_pipelined[i] = g[r][c];
	}

	// gather parallel pipelined floyds results at root node and print adjacency matrix.
	// also print all the results such as runtimes and the speedups since we are done executing 
	// the different variations of floyd's algorithm.
	if (pid == 0) {
		MPI_Gather(g_buff_pipelined, count, MPI_INT, G_buff_pipelined, count, MPI_INT, 0, MPI_COMM_WORLD);
		unflatten_after_gather(G, G_buff_pipelined, p, p_grid_width, dim);

		print_adj_mat(G, n, "cost matrix after parallel pipelined floyd's algorithm", n < PRINT_CUTOFF);
		print_program_results(n, p, Ts, Tp_bcast, Tp_pipelined);

		// cleanup
		free_2d_matrix(G, n);
		delete[] G_buff_pipelined;
	}
	else {
		MPI_Gather(g_buff_pipelined, count, MPI_INT, G_buff_pipelined, count, MPI_INT, 0, MPI_COMM_WORLD);
	}

	delete[] g_buff_pipelined;

	// cleanup
	free_2d_matrix(g, dim);

	MPI_Finalize();
}