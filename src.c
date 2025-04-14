#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <float.h>

#define INDEX(x, y, z, t, nx, ny, nz, nc) ((((z) * (ny) + (y)) * (nx) + (x)) * (nc) + (t))

int PX, PY, PZ, NX, NY, NZ, NC;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int size, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (argc != 10)
    {
        if (myrank == 0)
            printf("Usage: %s <file> PX PY PZ NX NY NZ NC <output>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // access command line arguments
    char *filename = argv[1];
    PX = atoi(argv[2]);
    PY = atoi(argv[3]);
    PZ = atoi(argv[4]);
    NX = atoi(argv[5]);
    NY = atoi(argv[6]);
    NZ = atoi(argv[7]);
    NC = atoi(argv[8]);
    char *out_file = argv[9];
    
    int Nprocess = PX * PY * PZ;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (Nprocess != size)
    {
        if (myrank == 0)
            printf("Error: PX*PY*PZ must equal number of processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Make subcommunicators
    int xrank, yrank, zrank, xyrank;
    MPI_Comm xcomm, ycomm, zcomm, xycomm;
    int zcolor = myrank / (PX * PY);
    MPI_Comm_split(MPI_COMM_WORLD, zcolor, myrank, &xycomm);
    MPI_Comm_rank(xycomm, &xyrank);
    MPI_Comm_split(xycomm, xyrank / PX, xyrank, &xcomm);
    MPI_Comm_rank(xcomm, &xrank);
    MPI_Comm_split(xycomm, xyrank % PX, xyrank, &ycomm);
    MPI_Comm_rank(ycomm, &yrank);
    MPI_Comm_split(MPI_COMM_WORLD, xyrank, myrank, &zcomm);
    MPI_Comm_rank(zcomm, &zrank);

    int LX = NX / PX, LY = NY / PY, LZ = NZ / PZ;
    int local_total = LX * LY * LZ;

    float **local_data = (float **)malloc(NC * sizeof(float *));
    for (int i = 0; i < NC; i++)
        local_data[i] = (float *)malloc(local_total * sizeof(float));

    int x0d = xrank * LX;
    int y0d = yrank * LY;
    int z0d = zrank * LZ;

    // Read the binary file in parallel
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Rank %d: Could not open binary file %s\n", myrank, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int lz = 0; lz < LZ; lz++)
    {
        int gz = z0d + lz;
        for (int ly = 0; ly < LY; ly++)
        {
            int gy = y0d + ly;

            size_t offset_in_floats =
                (size_t)gz * (NY * NX * NC) +
                (size_t)gy * (NX * NC) +
                (size_t)x0d * NC;

            fseek(fp, offset_in_floats * sizeof(float), SEEK_SET);

            size_t chunk_size = (size_t)LX * NC;
            float *buffer = (float *)malloc(chunk_size * sizeof(float));

            size_t nread = fread(buffer, sizeof(float), chunk_size, fp);
            if (nread != chunk_size)
            {
                printf("Rank %d: Error reading file at z=%d, y=%d. "
                       "Read %zu floats instead of %zu.\n",
                       myrank, gz, gy, nread, chunk_size);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (int lx = 0; lx < LX; lx++)
            {
                for (int c = 0; c < NC; c++)
                {
                    int idx = (lz * LY + ly) * LX + lx;
                    local_data[c][idx] = buffer[lx * NC + c];
                }
            }

            free(buffer);
        }
    }
    fclose(fp);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Make MPI derived data types of faces
    MPI_Datatype yz_face, xz_face, xy_face;
    MPI_Type_vector(LY * LZ, 1, LX, MPI_FLOAT, &yz_face);  // for X face
    MPI_Type_vector(LZ, LX, LX * LY, MPI_FLOAT, &xz_face); // for Y face
    MPI_Type_vector(LX * LY, 1, 1, MPI_FLOAT, &xy_face);   // for Z face
    MPI_Type_commit(&yz_face);
    MPI_Type_commit(&xz_face);
    MPI_Type_commit(&xy_face);

    MPI_Request reqs[12 * NC];
    int req_idx = 0;

    // Making recvbufs
    float **recvbuf_x[2], **recvbuf_y[2], **recvbuf_z[2];
    for (int d = 0; d < 2; d++)
    {
        recvbuf_x[d] = (float **)malloc(NC * sizeof(float *));
        recvbuf_y[d] = (float **)malloc(NC * sizeof(float *));
        recvbuf_z[d] = (float **)malloc(NC * sizeof(float *));
        for (int t = 0; t < NC; t++)
        {
            recvbuf_x[d][t] = (float *)malloc(LY * LZ * sizeof(float));
            recvbuf_y[d][t] = (float *)malloc(LX * LZ * sizeof(float));
            recvbuf_z[d][t] = (float *)malloc(LX * LY * sizeof(float));
        }
    }

    // Halo-exchange
    for (int t = 0; t < NC; t++)
    {
        float *block = local_data[t];

        if (xrank > 0)
        {
            MPI_Irecv(recvbuf_x[0][t], LY * LZ, MPI_FLOAT, xrank - 1, t, xcomm, &reqs[req_idx++]);
            MPI_Isend(&block[0], 1, yz_face, xrank - 1, t, xcomm, &reqs[req_idx++]);
        }
        if (xrank < PX - 1)
        {
            MPI_Irecv(recvbuf_x[1][t], LY * LZ, MPI_FLOAT, xrank + 1, t, xcomm, &reqs[req_idx++]);
            MPI_Isend(&block[LX - 1], 1, yz_face, xrank + 1, t, xcomm, &reqs[req_idx++]);
        }

        if (yrank > 0)
        {
            MPI_Irecv(recvbuf_y[0][t], LX * LZ, MPI_FLOAT, yrank - 1, t, ycomm, &reqs[req_idx++]);
            MPI_Isend(&block[0], 1, xz_face, yrank - 1, t, ycomm, &reqs[req_idx++]);
        }
        if (yrank < PY - 1)
        {
            MPI_Irecv(recvbuf_y[1][t], LX * LZ, MPI_FLOAT, yrank + 1, t, ycomm, &reqs[req_idx++]);
            MPI_Isend(&block[LX * (LY - 1)], 1, xz_face, yrank + 1, t, ycomm, &reqs[req_idx++]);
        }

        if (zrank > 0)
        {
            MPI_Irecv(recvbuf_z[0][t], LX * LY, MPI_FLOAT, zrank - 1, t, zcomm, &reqs[req_idx++]);
            MPI_Isend(&block[0], 1, xy_face, zrank - 1, t, zcomm, &reqs[req_idx++]);
        }
        if (zrank < PZ - 1)
        {
            MPI_Irecv(recvbuf_z[1][t], LX * LY, MPI_FLOAT, zrank + 1, t, zcomm, &reqs[req_idx++]);
            MPI_Isend(&block[LX * LY * (LZ - 1)], 1, xy_face, zrank + 1, t, zcomm, &reqs[req_idx++]);
        }
    }

    MPI_Waitall(req_idx, reqs, MPI_STATUSES_IGNORE);

    // Compute min/max counts
    int *min_count = (int *)calloc(NC, sizeof(int));
    int *max_count = (int *)calloc(NC, sizeof(int));
    float *gmin = (float *)malloc(NC * sizeof(float));
    float *gmax = (float *)malloc(NC * sizeof(float));

    for (int t = 0; t < NC; t++)
    {
        gmin[t] = +FLT_MAX;
        gmax[t] = -FLT_MAX;

        for (int lz = 0; lz < LZ; lz++)
        {
            for (int ly = 0; ly < LY; ly++)
            {
                for (int lx = 0; lx < LX; lx++)
                {
                    int idx = lz * LY * LX + ly * LX + lx;
                    float val = local_data[t][idx];
                    int is_min = 1, is_max = 1;

                    int dx[] = {1, -1, 0, 0, 0, 0};
                    int dy[] = {0, 0, 1, -1, 0, 0};
                    int dz[] = {0, 0, 0, 0, 1, -1};

                    for (int d = 0; d < 6; d++)
                    {
                        int nx = lx + dx[d];
                        int ny = ly + dy[d];
                        int nz = lz + dz[d];

                        float nval;

                        if (nx >= 0 && nx < LX &&
                            ny >= 0 && ny < LY &&
                            nz >= 0 && nz < LZ)
                        {
                            int nidx = nz * LY * LX + ny * LX + nx;
                            nval = local_data[t][nidx];
                        }
                        else
                        {
                            if (nx == -1 && xrank > 0)
                                nval = recvbuf_x[0][t][lz * LY + ly];
                            else if (nx == LX && xrank < PX - 1)
                                nval = recvbuf_x[1][t][lz * LY + ly];
                            else if (ny == -1 && yrank > 0)
                                nval = recvbuf_y[0][t][lz * LX + lx];
                            else if (ny == LY && yrank < PY - 1)
                                nval = recvbuf_y[1][t][lz * LX + lx];
                            else if (nz == -1 && zrank > 0)
                                nval = recvbuf_z[0][t][ly * LX + lx];
                            else if (nz == LZ && zrank < PZ - 1)
                                nval = recvbuf_z[1][t][ly * LX + lx];
                            else
                                continue;
                        }

                        if (val >= nval)
                            is_min = 0;
                        if (val <= nval)
                            is_max = 0;
                    }

                    if (is_min)
                        min_count[t]++;
                    if (is_max)
                        max_count[t]++;
                    if (val < gmin[t])
                        gmin[t] = val;
                    if (val > gmax[t])
                        gmax[t] = val;
                }
            }
        }
    }

    int *global_min_count = NULL;
    int *global_max_count = NULL;
    float *global_gmin = NULL;
    float *global_gmax = NULL;

    if (myrank == 0)
    {
        global_min_count = (int *)calloc(NC, sizeof(int));
        global_max_count = (int *)calloc(NC, sizeof(int));
        global_gmin = (float *)malloc(NC * sizeof(float));
        global_gmax = (float *)malloc(NC * sizeof(float));
    }

    // Reduce to rank 0 to get local min/max counts and global min/max values
    MPI_Reduce(min_count, global_min_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(max_count, global_max_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(gmin, global_gmin, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(gmax, global_gmax, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    if (myrank == 0)
    {
        FILE *fp_out = fopen(out_file, "w");
        if (!fp_out)
        {
            printf("Could not open output file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int t = 0; t < NC; t++)
        {
            fprintf(fp_out, "(%d,%d)", global_min_count[t], global_max_count[t]);
            if (t < NC - 1)
                fprintf(fp_out, ",");
        }
        fprintf(fp_out, "\n");

        for (int t = 0; t < NC; t++)
        {
            fprintf(fp_out, "(%.4f,%.4f)", global_gmin[t], global_gmax[t]);
            if (t < NC - 1)
                fprintf(fp_out, ",");
        }
        fprintf(fp_out, "\n");
        fclose(fp_out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t4 = MPI_Wtime();

    // Compute local times
    double local_readTime = t2 - t1;     // file read + distribution (now each rank reads itself)
    double local_mainCodeTime = t3 - t2; // main code
    double local_totalTime = t4 - t1;    // entire run

    double readTime, mainCodeTime, totalTime;
    MPI_Reduce(&local_readTime, &readTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_mainCodeTime, &mainCodeTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_totalTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myrank == 0)
    {
        FILE *fp_out = fopen(out_file, "a");
        if (!fp_out)
        {
            printf("Could not open output file for appending.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp_out, "%.6f %.6f %.6f\n", readTime, mainCodeTime, totalTime);
        fclose(fp_out);
    }

    // Cleanup
    free(min_count);
    free(max_count);
    free(gmin);
    free(gmax);

    if (myrank == 0)
    {
        free(global_min_count);
        free(global_max_count);
        free(global_gmin);
        free(global_gmax);
    }

    MPI_Type_free(&yz_face);
    MPI_Type_free(&xz_face);
    MPI_Type_free(&xy_face);

    for (int d = 0; d < 2; d++)
    {
        for (int t = 0; t < NC; t++)
        {
            free(recvbuf_x[d][t]);
            free(recvbuf_y[d][t]);
            free(recvbuf_z[d][t]);
        }
        free(recvbuf_x[d]);
        free(recvbuf_y[d]);
        free(recvbuf_z[d]);
    }

    for (int i = 0; i < NC; i++)
        free(local_data[i]);
    free(local_data);

    MPI_Finalize();
    return 0;
}
