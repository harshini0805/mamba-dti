import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FASTA_DIR, PSSM_DIR, DATASET

PSSM_DIR.mkdir(parents=True, exist_ok=True)

print(f"[DATASET={DATASET}] reading FASTA from {FASTA_DIR} -> writing PSSMs to {PSSM_DIR}")

PSIBLAST_PATH = r"C:\Program Files\NCBI\blast-2.17.0+\bin\psiblast.exe"
BLAST_DB = r"C:\blast_db\swissprot"


NUM_ITERATIONS = 3
EVALUE = 0.001
NUM_THREADS_PER_BLAST = 2
NUM_PARALLEL_PROCESSES = 4

# FALLBACK_EVALUE: used ONLY for proteins that get a genuine
# "***** No hits found *****" against EVALUE=0.001 (short/divergent
# sequences with no confident SwissProt homolog at that strict cutoff —
# same failure mode hit on celegans_random/bindingdb_random). Set to 10^3,
# matching the precedent for this exact situation (a very short, e.g.
# 23-residue, target with no hits at the strict threshold gets the
# threshold relaxed to 10^3 for that one protein only, everything else
# stays at 10^-3). It is NOT applied to every protein, and which ones it
# was used for is logged to pssm_evalue_used.csv so this never silently
# mixes high- and low-confidence PSSMs without a record of which is which.
FALLBACK_EVALUE = 1e3



fasta_files = sorted(FASTA_DIR.glob("*.fasta"))

print(f"Found {len(fasta_files)} FASTA files.")



def run_psiblast(fasta_path: Path, output_pssm: Path, evalue: float):

    cmd = [
        PSIBLAST_PATH,
        "-query", str(fasta_path),
        "-db", BLAST_DB,
        "-num_iterations", str(NUM_ITERATIONS),
        "-evalue", str(evalue),
        "-num_threads", str(NUM_THREADS_PER_BLAST),
        "-out_ascii_pssm", str(output_pssm)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # "Succeeded" means: exited 0 AND produced a non-empty file. A 0-byte
    # file (psiblast's "No hits found" case) is cleaned up immediately so
    # it's never mistaken for a completed PSSM on a later rerun.
    got_output = output_pssm.exists() and output_pssm.stat().st_size > 0

    if output_pssm.exists() and output_pssm.stat().st_size == 0:
        output_pssm.unlink()

    ok = (result.returncode == 0) and got_output

    return ok, result


def generate_pssm(fasta_path: Path):

    protein_name = fasta_path.stem

    output_pssm = PSSM_DIR / f"{protein_name}.pssm"

    # Skip already generated PSSMs. Checks size too, not just existence —
    # a prior failed run could in principle leave a 0-byte file behind, and
    # treating that as "done" would mean it can never be retried.
    if output_pssm.exists() and output_pssm.stat().st_size > 0:
        return protein_name, "SKIPPED", None, None

    try:

        ok, result = run_psiblast(fasta_path, output_pssm, EVALUE)

        if ok:
            return protein_name, "SUCCESS", EVALUE, None

        primary_reason = (
            f"No hits at EVALUE={EVALUE} (returncode={result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        # Retry once, only for this protein, with the relaxed fallback
        # e-value — recovers coverage for short/divergent sequences that
        # have no confident hit at the strict default.
        ok2, result2 = run_psiblast(fasta_path, output_pssm, FALLBACK_EVALUE)

        if ok2:
            return protein_name, "SUCCESS", FALLBACK_EVALUE, None

        reason = (
            f"{primary_reason}\n\n"
            f"---- retry at FALLBACK_EVALUE={FALLBACK_EVALUE} also failed ----\n"
            f"returncode={result2.returncode}\n"
            f"stdout:\n{result2.stdout}\nstderr:\n{result2.stderr}"
        )
        return protein_name, "FAILED", None, reason

    except Exception as exc:
        return protein_name, "FAILED", None, f"Python exception: {exc!r}"



if __name__ == "__main__":

    print("Generating PSSM Files")

    with Pool(processes=NUM_PARALLEL_PROCESSES) as pool:

        results = list(
            tqdm(
                pool.imap(generate_pssm, fasta_files),
                total=len(fasta_files),
                desc="Generating PSSMs"
            )
        )

    success = sum(status == "SUCCESS" for _, status, _, _ in results)
    skipped = sum(status == "SKIPPED" for _, status, _, _ in results)
    failed = sum(status == "FAILED" for _, status, _, _ in results)
    fallback_used = sum(
        status == "SUCCESS" and evalue == FALLBACK_EVALUE
        for _, status, evalue, _ in results
    )

    print("\n")

    print("SUMMARY")


    print(f"Total FASTA Files : {len(fasta_files)}")
    print(f"Generated         : {success}  ({fallback_used} of these needed the relaxed FALLBACK_EVALUE={FALLBACK_EVALUE})")
    print(f"Skipped           : {skipped}")
    print(f"Failed            : {failed}")

    # Save failed proteins (names only, same as before — used by other
    # scripts/manual inspection that just want the list)
    failed_proteins = [
        protein
        for protein, status, _, _ in results
        if status == "FAILED"
    ]

    failed_log = PSSM_DIR / "failed_proteins.txt"

    with open(failed_log, "w") as f:
        for protein in failed_proteins:
            f.write(protein + "\n")

    print(f"\nFailure log saved to:\n{failed_log}")

    # Save WHY each one failed (psiblast stdout/stderr) — previously
    # discarded entirely, which made a repeat failure undiagnosable.
    if failed_proteins:

        errors_log = PSSM_DIR / "failed_pssm_errors.txt"

        with open(errors_log, "w") as f:
            for protein, status, _, reason in results:
                if status == "FAILED":
                    f.write(f"{'=' * 60}\n{protein}\n{'=' * 60}\n{reason}\n\n")

        print(f"Per-protein failure reasons saved to:\n{errors_log}")

    # Save which proteins only succeeded via the relaxed fallback e-value —
    # these PSSMs rest on much weaker homology evidence than everything
    # else, so this is the one place that's on record, not silently mixed
    # into the rest of the (mostly EVALUE=0.001) PSSMs.
    fallback_proteins = [
        (protein, evalue)
        for protein, status, evalue, _ in results
        if status == "SUCCESS" and evalue == FALLBACK_EVALUE
    ]

    if fallback_proteins:

        fallback_log = PSSM_DIR / "pssm_evalue_used.csv"

        with open(fallback_log, "w") as f:
            f.write("protein,evalue_used\n")
            for protein, evalue in fallback_proteins:
                f.write(f"{protein},{evalue}\n")

        print(f"Proteins generated via relaxed EVALUE logged to:\n{fallback_log}")

    print("\nDone!")
