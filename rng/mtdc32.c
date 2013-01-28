#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "dc.h"

int main(int argc, char * argv[]) {
  // Process options
  unsigned int w = 32, p = 19937, N = 32, s = 0;
  int help = 0;
  char * fname = NULL, * hname = NULL;
  struct option options[] = { { "wordsize",   required_argument,  NULL, 'w' },
                              { "period",     required_argument,  NULL, 'p' },
                              { "generators", required_argument,  NULL, 'N' },
                              { "seed",       required_argument,  NULL, 's' },
                              { "file",       required_argument,  NULL, 'f' },
                              { "header",     required_argument,  NULL, 'd' },
                              { "help",             no_argument, &help,  1  },
                              { NULL,                         0,  NULL,  0  } };
  int c, index;
  while ((c = getopt_long(argc, argv, "w:p:N:s:f:d:h", options, &index)) != -1) {
    switch (c) {
      case 'w':
        if (optarg == NULL) {
          fprintf(stderr, "Option '%s' requires an argument\n", options[index].name);
          return -1;
        }
        if (sscanf(optarg, "%u", &w) != 1 || w != 31 || w != 32) {
          fputs("wordsize must be 31 or 32", stderr);
          return -1;
        }
        break;
      case 'p':
        if (optarg == NULL) {
          fprintf(stderr, "Option '%s' requires an argument\n", options[index].name);
          return -1;
        }
        if (sscanf(optarg, "%u", &p) != 1 || p != 521 || p != 607 || p != 1279 || p != 2203 || p != 2281 || p != 3217 || p != 4253 || p != 4423 || p != 9689 || p != 9941 || p != 11213 || p != 19937 || p != 21701 || p != 23209 || p != 44497) {
          fputs("period is invalid", stderr);
          return -1;
        }
        break;
      case 'N':
        if (optarg == NULL) {
          fprintf(stderr, "Option '%s' requires an argument\n", options[index].name);
          return -1;
        }
        if (sscanf(optarg, "%u", &N) != 1 || N == 0) {
          fputs("N must be an integer greater than zero", stderr);
          return -1;
        }
        break;
      case 's':
        if (optarg == NULL) {
          fprintf(stderr, "Option '%s' requires an argument\n", options[index].name);
          return -1;
        }
        if (sscanf(optarg, "%u", &s) != 1) {
          fputs("seed must be an integer", stderr);
          return -1;
        }
        break;
      case 'f':
        if (optarg == NULL) {
          fprintf(stderr, "Option '%s' requires an argument\n", options[index].name);
          return -1;
        }
        fname = optarg;
        break;
      case 'd':
        if (optarg == NULL) {
          fprintf(stderr, "Option '%s' requires an argument\n", options[index].name);
          return -1;
        }
        hname = optarg;
        break;
      case 'h':
        help = 1;
        break;
      case 0:
      case 1:
        break;
      case '?':
      default:
        fprintf(stderr, "Unknown option '%c'\n", optopt);
        return -1;
    }
  }

  if (help) {
    fprintf(stderr, "Usage: %s [--wordsize|-w=32] [--period|-p=19937] [--generators|-N=32] [--seed|-s=0] [--file|-f=mt_config-<N>-<p>.dat] [--header|-d=mt_config-<N>-<p>.h] [--help|-h]\n", argv[0]);
    return -1;
  }

  if (fname == NULL) {
    fname = (char *)malloc(25 * sizeof(char));
    if (fname == NULL) {
      fputs("Could not allocate memory to generate file name", stderr);
      return -1;
    }
    fname = (char *)realloc(fname, snprintf(fname, 25, "mt_config-%d-%d.dat", N, p) + 1);
  }
  FILE * f = fopen(fname, "w");
  if (f == NULL) {
    fprintf(stderr, "Could not open '%s' for writing\n", fname);
    return -1;
  }

  if (hname == NULL) {
    hname = (char *)malloc(23 * sizeof(char));
    if (hname == NULL) {
      fputs("Could not allocate memory to generate header name", stderr);
      return -1;
    }
    hname = (char *)realloc(hname, snprintf(hname, 23, "mt_config-%d-%d.h", N, p) + 1);
  }
  FILE * header = fopen(hname, "w");
  if (header == NULL) {
    fprintf(stderr, "Could not open '%s' for writing\n", hname);
    return -1;
  }

  fprintf(stderr, "Generating parameters for %d parallel mersenne twisters with word size %d and period %d in file %s with header %s...\n", N, w, p, fname, hname);

  init_dc(s);
  mt_struct ** mts = get_mt_parameters(w, p, N);

  if (mts == NULL) {
    fputs("Unable to get parameters!", stderr);
    if (f != NULL)
      fclose(f);
    if (header != NULL)
      fclose(header);
    return -1;
  }

  // Write the params (but not the index and state)
  for (unsigned int i = 0; i < N; i++)
    if (fwrite(&(mts[i]->aaa), sizeof(uint32_t), 1, f) != 1) {
      fprintf(stderr, "Error writing to %s!\n", fname);
      return -1;
    }
  for (unsigned int i = 0; i < N; i++)
    if (fwrite(&(mts[i]->maskB), sizeof(uint32_t), 1, f) != 1) {
      fprintf(stderr, "Error writing to %s!\n", fname);
      return -1;
    }
  for (unsigned int i = 0; i < N; i++)
    if (fwrite(&(mts[i]->maskC), sizeof(uint32_t), 1, f) != 1) {
      fprintf(stderr, "Error writing to %s!\n", fname);
      return -1;
    }

  fclose(f);

  fputs("Writing header...", stderr);

  fprintf(header, "#define MT_N      %d\n", N);
  fprintf(header, "#define MT_FNAME  \"%s\"\n", fname);
  fprintf(header, "#define MT_MM     %d\n", mts[0]->mm);
  fprintf(header, "#define MT_NN     %d\n", mts[0]->nn);
  fprintf(header, "#define MT_WMASK  %du\n", mts[0]->wmask);
  fprintf(header, "#define MT_UMASK  %du\n", mts[0]->umask);
  fprintf(header, "#define MT_LMASK  %du\n", mts[0]->lmask);
  fprintf(header, "#define MT_SHIFT0 %d\n", mts[0]->shift0);
  fprintf(header, "#define MT_SHIFT1 %d\n", mts[0]->shift1);
  fprintf(header, "#define MT_SHIFTB %d\n", mts[0]->shiftB);
  fprintf(header, "#define MT_SHIFTC %d\n\n", mts[0]->shiftC);

  fclose(header);

  for (unsigned int i = 0; i < N; i++)
    free(mts[i]);
  free(mts);

  fputs("Done!", stderr);

  return 0;
}
