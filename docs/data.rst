Example datasets
================

The manuscript evaluates Tracel on five human datasets from three platforms and three tissues. Counts below are
the initial spatial cells and evaluation genes after input preparation.

.. list-table::
   :header-rows: 1
   :widths: 16 12 12 30 30

   * - Dataset
     - Platform
     - Tissue
     - Spatial data
     - scRNA-seq reference
   * - Sample2
     - Xenium
     - Breast cancer
     - 10x Xenium FFPE human breast cancer preview, Sample 2; 142,272 cells, 275 genes; annotations from
       Janesick et al. (2023), Fig. 6 (unsupervised)
     - SCP1039 (Single Cell Portal), ``celltype_minor``
   * - Janesick2023
     - Xenium
     - Breast cancer
     - Xenium Rep1 of Janesick et al. (2023); 167,780 cells, 313 genes; supervised annotations
     - 27,472 annotated scFFPE-seq cells from the same study
   * - CosMx lung
     - CosMx
     - Lung cancer
     - Lung5 Rep2 (NanoString public NSCLC data); 105,800 cells, 951 genes; Giotto annotations;
       coordinates converted at 0.18 µm per pixel
     - GSE131907 tumor-lung cells
   * - Xenium lung
     - Xenium
     - Lung cancer
     - 10x Xenium human lung cancer FFPE; contiguous region of 49,786 cells, 374 genes; official graph clusters
       annotated with markers; transcripts with qv < 20 removed
     - GSE131907 tumor-lung cells
   * - P2CRC
     - Visium HD
     - Colorectal cancer
     - Visium HD colorectal cancer sample P2 (Oliveira et al., 2025); 49,927 StarDist H&E nuclei;
       OT2 on 2,000 OT1-selected genes
     - P2CRC scRNA-seq reference (Oliveira et al., 2025)

Sources
-------

* 10x Genomics datasets: https://www.10xgenomics.com/datasets
* Janesick et al., *Nature Communications* (2023): https://www.nature.com/articles/s41467-023-43458-x
* SCP1039: https://singlecell.broadinstitute.org/single_cell/study/SCP1039
* GSE131907: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907
* Oliveira et al., high-definition spatial transcriptomic profiling of colorectal cancer (2025)

``examples/sample2/make_inputs.py`` shows how the Sample2 inputs were assembled.
