# Full FlyWire FAFB v783 connectome edge list

**Research date:** 2026-09-23

## Bottom line

**Naming note:** official sources call this the **FlyWire FAFB v783 / Codex** release. The export filename contains `princeton` and the official object is hosted in Google Cloud Storage, but I found no primary source using “Google/Princeton release” as a separate dataset name.

For the **full, no-analysis-threshold neuron-level edge export currently served by Codex**, download:

- **File:** `connections_princeton_no_threshold.csv.gz`
- **Direct, current GCS URL:** <https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections_princeton_no_threshold.csv.gz>
- **Generation-pinned URL:** <https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections_princeton_no_threshold.csv.gz?generation=1751983882423412>
- **Size:** **275,679,780 bytes** = **275.680 MB** (decimal) = **262.909 MiB**
- **Format:** **gzip-compressed CSV**, not Parquet
- **MD5:** `694f7e5bd018b83c71eeb0ba55b50e7e`
- **Auth using the direct GCS object:** none observed; an unauthenticated `HEAD` request returned `200 OK` on the research date.

This is the correct file to choose when “full” means **all proofread FAFB v783 neuron-neuron-neuropil connections with at least one synapse**, without applying the paper/UI's minimum-weight edge threshold. Its verified header is:

```text
pre_root_id,post_root_id,neuropil,syn_count,nt_type
```

Thus it is an aggregated directed edge list, not a per-synapse table: one row represents a presynaptic root ID, postsynaptic root ID, and neuropil, with `syn_count` as the edge weight. The file name explicitly says `no_threshold`, and a small range read verified rows with `syn_count=1`. The official GCS object metadata reports the size, content type, generation, and hashes here: [Google Cloud Storage object metadata](https://www.googleapis.com/storage/v1/b/flywire-data/o/codex%2Fdata%2Ffafb%2F783%2Fconnections_princeton_no_threshold.csv.gz). The full public bucket prefix is visible in the [official `flywire-data` listing](https://storage.googleapis.com/flywire-data/?prefix=codex/data/fafb/783/).

## Download command

This object is below 500 MB:

```bash
BASE='https://storage.googleapis.com/flywire-data/codex/data/fafb/783'
curl -fL --retry 3 -C - \
  -o connections_princeton_no_threshold.csv.gz \
  "$BASE/connections_princeton_no_threshold.csv.gz"
```

Check its size and checksum without downloading it:

```bash
URL='https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections_princeton_no_threshold.csv.gz'
curl -fsSI "$URL"
md5sum connections_princeton_no_threshold.csv.gz
# expected: 694f7e5bd018b83c71eeb0ba55b50e7e
```

No file was downloaded during this research; only HTTP metadata and small range requests were used.

## Sign-in and authentication

There are two access paths, with different current behavior:

1. **Direct public GCS object:** no Google login, Codex token, or CAVE token was needed in an unauthenticated request on 2026-09-23. This is an observation about current bucket policy, not a guarantee about future access controls.
2. **Codex download portal:** use [`https://codex.flywire.ai/api/download?dataset=fafb`](https://codex.flywire.ai/api/download?dataset=fafb). The unauthenticated page presents Google sign-in. Codex says its interactive apps require Google sign-in, while static public pages do not ([Codex home](https://codex.flywire.ai/)). The current programmatic resource route requires a **Codex API token**, copied from the Codex account page. An unauthenticated request returned:

   ```json
   {"error":"Missing or invalid api_token. Copy your token from Codex account page."}
   ```

   The documented route pattern is:

   ```text
   https://codex.flywire.ai/api/download_resource?data_product=PRODUCT&dataset=fafb&api_token=TOKEN
   ```

   For the no-threshold edge export, the corresponding product/file name is `connections_princeton_no_threshold`:

   ```bash
   curl -fL --retry 3 \
     -o connections_princeton_no_threshold.csv.gz \
     'https://codex.flywire.ai/api/download_resource?data_product=connections_princeton_no_threshold&dataset=fafb&api_token=YOUR_CODEX_TOKEN'
   ```

   Treat the token as a secret because query-string credentials can enter shell history or logs. This is a **Codex API token**, not a CAVE token. The [official Codex FAQ](https://codex.flywire.ai/faq) documents the static-download API pattern and says bulk users should use the static files rather than live queries.

## Cell types and neurotransmitter predictions

### Included directly in the edge file

The no-threshold edge CSV has an `nt_type` column containing the predicted neurotransmitter label for each aggregated connection. It does **not** embed cell-type annotations and it does not contain the confidence/probability columns in this compact CSV.

### Cell-type sidecar

Download the current consolidated cell-type table from the same v783 prefix:

- **URL:** <https://storage.googleapis.com/flywire-data/codex/data/fafb/783/consolidated_cell_types.csv.gz>
- **Size:** **901,707 bytes** (0.902 MB / 0.860 MiB)
- **Format:** gzip-compressed CSV
- **Header:** `root_id,primary_type,additional_type(s)`
- **Join:** map `edge.pre_root_id` and `edge.post_root_id` to `cell_types.root_id` to label both endpoints.
- **Metadata:** [GCS object metadata](https://www.googleapis.com/storage/v1/b/flywire-data/o/codex%2Fdata%2Ffafb%2F783%2Fconsolidated_cell_types.csv.gz)

This is a separate sidecar, not a column in the full edge list. The current Codex FAQ says Codex cell typing is consolidated from multiple sources, including the flagship FlyWire and optic-lobe papers, and recommends the static CSV for bulk analysis.

### Neurotransmitter score/probability sidecar

For neuron-level predicted transmitter identity, confidence, and class probabilities, also download:

- **URL:** <https://storage.googleapis.com/flywire-data/codex/data/fafb/783/cell_stats.csv.gz>
- **Size:** **2,526,548 bytes** (2.527 MB / 2.410 MiB)
- **Format:** gzip-compressed CSV
- **Verified header:** `root_id,group,nt_type,nt_type_score,da_avg,ser_avg,gaba_avg,glut_avg,ach_avg,oct_avg`
- **Join:** again on root ID, for the pre- and/or post-synaptic neuron.
- **Metadata:** [GCS object metadata](https://www.googleapis.com/storage/v1/b/flywire-data/o/codex%2Fdata%2Ffafb%2F783%2Fcell_stats.csv.gz)

The compact edge table's `nt_type` and this neuron-level table's `nt_type` are predictions, not universally verified experimental labels. The Codex FAQ distinguishes predicted fields from curated/verified neurotransmitter fields.

Optional hierarchical annotations are in [`classification.csv.gz`](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/classification.csv.gz), **934,402 bytes**, with header `root_id,flow,super_class,class,sub_class,hemilineage,side,nerve`.

### Publication-pinned annotations

If the analysis must align with the 2024 Schlegel/Dorkenwald papers rather than the current Codex annotation mix, the FlyConnectome annotation repository's [`v2.1.0`](https://github.com/flyconnectome/flywire_annotations/releases/tag/v2.1.0) release says it is the version reported in both Nature papers. Its neuron annotation table is:

- **URL:** <https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v2.1.0/supplemental_files/Supplemental_file1_neuron_annotations.tsv>
- **Size:** **27,015,208 bytes** (27.015 MB / 25.764 MiB), per the [tagged GitHub file metadata](https://api.github.com/repos/flyconnectome/flywire_annotations/contents/supplemental_files/Supplemental_file1_neuron_annotations.tsv?ref=v2.1.0)
- **Format:** uncompressed TSV
- **Relevant columns:** `root_id`, `flow`, `super_class`, `cell_class`, `cell_sub_class`, `cell_type`, `hemibrain_type`, `top_nt`, `top_nt_conf`, `known_nt`, `known_nt_source`, `side`, and `nerve`.

The repository warns that some, but not all, of these annotations appear in Codex and that the two presentations can differ; use this tagged table when exact paper-era annotation provenance matters ([repository README](https://github.com/flyconnectome/flywire_annotations)).

## Thresholding: “full” versus the paper's displayed graph

Dorkenwald et al. report **139,255 proofread neurons and 54.5 million synapses between them**. The paper used a **five-synapse threshold** to decide whether two neurons were connected in the analyses and in the connections displayed in Codex, because lower-count edges are more likely to contain errors. Schlegel et al. describe the complete graph as about **15.1 million weighted edges** ([Dorkenwald et al., Nature 2024](https://www.nature.com/articles/s41586-024-07558-y); [Schlegel et al., Nature 2024](https://www.nature.com/articles/s41586-024-07686-5)).

Therefore:

- Use `connections_princeton_no_threshold.csv.gz` to obtain the **full data**, then apply `syn_count >= 5` when reproducing the paper/Codex thresholded graph.
- Do not mistake the 68,456,801-byte [`connections_princeton.csv.gz`](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections_princeton.csv.gz) filtered export for the requested no-threshold file. The [Codex FAQ](https://codex.flywire.ai/faq) lists five synapses as the default FAFB connectivity threshold.
- The homepage's current **3,732,460 connections** is a thresholded summary count, not evidence that the no-threshold edge list has only that many rows.

## Exact 2024 publication archive versus current Codex export

There are two official, non-interchangeable downloads:

| Purpose | Official artifact | Format | Exact size | Auth | Cell types included? |
|---|---|---:|---:|---|---|
| Current practical full Codex v783 edge export | [`connections_princeton_no_threshold.csv.gz`](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections_princeton_no_threshold.csv.gz) | gzip CSV | 275,679,780 bytes | Direct GCS currently none; portal token required | No; separate `consolidated_cell_types.csv.gz` |
| Publication-pinned paper-era proofread edge list | [`proofread_connections_783.feather`](https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content) | Apache Arrow **Feather**, not CSV/Parquet | 852,022,274 bytes (852.022 MB / 812.552 MiB) | None; Zenodo record is open (CC BY 4.0) | No |

The 2024 Zenodo record is the official **FlyWire Whole-brain Connectome Connectivity Data, version 783.0**, deposited by the FlyWire Consortium/Princeton. Its metadata says `proofread_connections_783.feather` is the proofread subset of the full synapse table, aggregated to one row per neuron-neuron-neuropil combination whenever there is at least one synapse. It contains `pre_pt_root_id`, `post_pt_root_id`, `neuropil`, `syn_count`, and the six average neurotransmitter probabilities (`gaba_avg`, `ach_avg`, `glut_avg`, `oct_avg`, `ser_avg`, `da_avg`). See the [Zenodo record](https://zenodo.org/records/10676866) and [machine-readable Zenodo API record](https://zenodo.org/api/records/10676866).

The Feather file is **over 500 MB**, so it was **not downloaded** during this task. Use it only when byte-level provenance to the 2024 release matters. The Codex FAQ also warns that files in the current portal are synchronized with current Codex data and may differ from original publication-time archives. The current GCS no-threshold object itself was generated on 2025-07-08, while some annotation sidecars were updated later; use the generation-pinned URL and record checksums when reproducibility matters.

## Files that are not the requested edge list

- [`fafb_v783_princeton_synapse_table.csv.gz`](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/fafb_v783_princeton_synapse_table.csv.gz) — **2,695,106,039 bytes**, gzip CSV. This is a per-synapse table, not a neuron-neuron edge list, and exceeds the 500 MB limit. It was not downloaded.
- [`flywire_synapses_783.feather`](https://zenodo.org/api/records/10676866/files/flywire_synapses_783.feather/content) — **9,492,998,242 bytes**. The Zenodo record describes this as roughly 130 million detected synapses in the whole EM volume, including synapses not attached to proofread neurons. It is not the proofread connectome edge list and was not downloaded.
- Skeleton/SWC and morphology downloads are neuron reconstructions, not connectivity edges.

## Dataset identity and citation

The requested dataset is **FlyWire FAFB v783**, the Female Adult Fly Brain connectome associated with:

- Dorkenwald et al., “Neuronal wiring diagram of an adult brain,” *Nature* 634, 124–138 (2024), DOI: <https://doi.org/10.1038/s41586-024-07558-y>.
- Schlegel et al., “Whole-brain annotation and multi-connectome cell typing of Drosophila,” *Nature* (2024), DOI: <https://doi.org/10.1038/s41586-024-07686-5>.
- Eckstein et al., “Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster,” *Cell* (2024), DOI: <https://doi.org/10.1016/j.cell.2024.03.016>.

The official [FlyWire overview](https://flywire.ai/) identifies the October 2024 flagship paper as containing 139,255 proofread neurons and links Codex as the data portal.
