Pipeline Corrections Based on Overview:

General:
- Change the Display Configuration Banner for some more user details. The steps should describe the modules too. For instance, error correct (ONT, PacBio) -> graph assembly (with EdgeWarden, PathWeaver) -> string graph (with ThreadCompass) -> Phasing (with Haplotype Detangler) [plus 2 iterations] -> graph cleanup -> finish, or error correct (ONT, PacBio) -> Illumina contigger -> graph assembly (with No AI/ML) -> Phasing (with Haplotype Detangler) [plus 3 iterations] -> graph cleanup (with SVScribe) -> finish


Specific Options:
- Step 4: Copy Hi-C data should be taken out of the pipeline, this isn't necessary.
- Step 9: Are all reads loaded into memory? Or just loaded as variables? It is impractical to load this amount of reads into memory.
- Step 11: ErrorProfiler should use the k-mer values from KWeaver for the frequency table.
- Step 16 and Step 26: Users are able to choose individual ai modules, so there should be more specific loading.
- Step 27: We should do this differently. If there is Illumina data, we follow that pipeline. If there isn't, we skip the contigger.
- Step 28: This is one of several times reads are combined or separated, we should minimize this.
- Step 29: Preparing Hi-C data is out of place here.
- Step 34: Contig extraction should not occur prior to Hi-C, ideally we want it to be applied to the graph...
- Step 40: This should be T2T-Polish
- Step 41: Check this algorithm, I don't think we wrote any gap filling.
Step 42: Expand the QC report


Pipeline Corrections:
- The KWeaver module must be run prior to error profiling, and k-mer values set for the length of the pipline (for error correction per read type, DBG, UL overlaps, etc.)
- The DBG modules are out of order. It should be DBG engine -> EdgeWarden -> PathWeaver, and no Haplotype Detangler until after the end of the first graph assembly iteration.
- After Hi-C scaffolding, Haplotype Detangler should be run so that we are now working with a phased graph (with the number of edges between nodes being somewhere between approximately 1 and the haplotypes input), then iterate the AI modules on the graph. After the final iteration, run SVScribe.
- We need to output some more stuff for the user. Final graph output, PathWeaver final scores, corrected reads coverage, UL coverage, etc. for Bandage NG
- The full assembly core (assuming the presence of long reads, ultra long reads, and Hi-C) is DBG engine -> EdgeWarden -> PathWeaver -> string graph engine -> ThreadCompass -> Hi-C scaffolding -> Haplotype Detangler -> (iteration cycle of EdgeWarden -> PathWeaver -> string graph engine -> ThreadCompass by user specified number of iterations) -> SVScribe -> finalize