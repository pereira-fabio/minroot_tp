/*
  This file simply creates some smooth values to test the speedup
  achived by the attacks against minroot
 */
#include <errno.h> // error codes
#include <stdio.h>
#include <stdlib.h> // convert strings to int/longs
#include <string.h> // to parse strings
#include <time.h>

#include <gmp.h>

// needed for reproducebility
#define RANDSEED 12380174892374

void print_usage()
{
	printf("Usage:\n");
	printf("minroottest [num challenges] [modulo size] [smothness parameter]\n");
}

// ENTRYPOINT
int main(int argc, char **argv)
{

	gmp_randstate_t randstate;
	mp_bitcnt_t bitbound, bittarget;
	unsigned int num_outputs;
	mpz_t result, temp, modulo, exponent;
	FILE *fptr = NULL;
	char savefolder[64];
	char buf[256];
	int count;
	clock_t start, end;

	if (argc < 4)
	{
		printf("Not enough arguments!\n");
		print_usage();
		return EINVAL;
	}
	else if (argc > 5)
	{
		printf("Too many arguments!\n");
		print_usage();
		return EINVAL;
	}

	num_outputs = strtoul(argv[1], (char **)NULL, 10);
	bittarget = strtoul(argv[2], (char **)NULL, 10);
	bitbound = strtoul(argv[3], (char **)NULL, 10);

	/* Build output path: data/challenges/k<bitbound>/ */
	snprintf(savefolder, sizeof(savefolder), "data/challenges/k%lu/", bitbound);
	{
		char mkdircmd[128];
		snprintf(mkdircmd, sizeof(mkdircmd), "mkdir -p %s", savefolder);
		if (system(mkdircmd) != 0)
			printf("Warning: could not create directory %s\n", savefolder);
	}

	/* Sanity check: warn if bitbound is larger than bittarget.
	   This is valid (one large prime exceeds target), just unusual. */
	if (bitbound > bittarget)
	{
		printf("Note: bitbound (%lu) > bittarget (%lu). "
			   "One prime will likely suffice.\n",
			   bitbound, bittarget);
	}

	printf("Output directory: %s\n", savefolder);
	printf("Finding %u numbers and one modulo of roughly %lu bits using %lu-bit primes\n", num_outputs, bittarget, bitbound);

	// initialise gmp random state
	gmp_randinit_default(randstate);
	gmp_randseed_ui(randstate, RANDSEED);

	// initialise int
	mpz_inits(result, temp, modulo, exponent, NULL);

	// find a modulo!
	mpz_set_ui(modulo, 1);
	mpz_mul_2exp(modulo, modulo, bittarget);
	mpz_nextprime(modulo, modulo);
	while (mpz_tdiv_ui(modulo, 3) != 2)
	{
		mpz_nextprime(modulo, modulo);
	}

	printf("Found suitable modulo of %lu bits\n", mpz_sizeinbase(modulo, 2));
	snprintf(buf, sizeof(buf), "%smodulo_%lu_%lu.txt", savefolder, bitbound, bittarget);
	fptr = fopen(buf, "wb");
	if (fptr == NULL)
	{
		printf("Error opening %s\n", buf);
	}
	else
	{
		mpz_out_raw(fptr, modulo);
		fclose(fptr);
		fptr = NULL;
		printf("Wrote to %s\n", buf);
	}

	// now construct challenges
	for (int i = 0; i < num_outputs; ++i)
	{
		// first clear the factors file
		snprintf(buf, sizeof(buf), "%sfactors_%u_%lu_%lu.txt", savefolder, i, bitbound, bittarget);
		fptr = fopen(buf, "w");
		if (fptr == NULL)
		{
			printf("Error clearing %s file\n", buf);
		}
		else
		{
			fclose(fptr);
			fptr = NULL;
		}

		snprintf(buf, sizeof(buf), "%sfactors_%u_%lu_%lu.txt", savefolder, i, bitbound, bittarget);
		printf("Saving factors for %i challenge to %s\n", i, buf);

		/* Compute how many bits we need before stopping.
		   bittarget - bitbound would underflow (unsigned) when bitbound >= bittarget,
		   causing an infinite loop. Use a signed comparison to guard against this. */
		count = 0;
		mpz_set_ui(result, 1);
		long bit_threshold = (long)bittarget - (long)bitbound;
		if (bit_threshold < 0)
			bit_threshold = 0;
		/* Always include at least one prime so the challenge is non-trivial */
		do
		{
			mpz_urandomb(temp, randstate, bitbound);
			mpz_nextprime(temp, temp);

			// make sure that multiplying by temp does not exceed the modulo
			if (mpz_sizeinbase(result, 2) + mpz_sizeinbase(temp, 2) > mpz_sizeinbase(modulo, 2))
			{
				continue;
			}

			fptr = fopen(buf, "ab");
			if (fptr == NULL)
			{
				printf("Error opening %s file\n", buf);
			}
			else
			{
				mpz_out_raw(fptr, temp);
				fclose(fptr);
				fptr = NULL;
			}

			mpz_mul(result, result, temp);
			++count;
		} while ((long)mpz_sizeinbase(result, 2) < bit_threshold);

		printf("Obtained a value with %lu bits through multiplications of %u primes\n", mpz_sizeinbase(result, 2), count);

		// now save the result
		snprintf(buf, sizeof(buf), "%schallenge_%u_%lu_%lu.txt", savefolder, i, bitbound, bittarget);
		fptr = fopen(buf, "wb");
		if (fptr == NULL)
		{
			printf("Error opening %s\n", buf);
		}
		else
		{
			mpz_out_raw(fptr, result);
			fclose(fptr);
			fptr = NULL;
			printf("Wrote to %s\n", buf);
		}

		// now let's do a quick timing test
		mpz_set_ui(temp, 3);
		mpz_sub_ui(modulo, modulo, 1);
		mpz_invert(exponent, temp, modulo);
		mpz_add_ui(modulo, modulo, 1);

		// cube
		mpz_powm_ui(temp, result, 3, modulo);

		// cuberoot
		start = clock();
		mpz_powm(temp, temp, exponent, modulo);
		end = clock();

		if (mpz_cmp(temp, result) != 0)
		{
			printf("Cuberoot not working!\n");
		}
		printf("\nTARGET TIME TO BEAT: %.3fms\n\n", 1000 * (double)(end - start) / CLOCKS_PER_SEC);
	}

	mpz_clears(result, temp, modulo, exponent, NULL);

	return 0;
}