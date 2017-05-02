-- This script is mainly reponsible for the following things
-- 1. extract cnn features and restore if needed.
-- 2. calculate 5 variables:
--  Because it contains many style layers, so for example kernels is a table containing 5 kernel.
--  And each kernel is the C*cluster_num.
--    kernels
--    counts:  Each kernel has how many points
--    S_alls:  It is a tensor containing all pixels in one style layer, and the value of it indicates its belonging cluster_index
--    is_nil_centroids: set 0 at the very begining, traversal to get non-pointed cluster


-- The second stage is to calculate the gram of each cluster
-- expand_style_features : mirror and zero
-- gram_cluster_all: According to S_alls to get the gram forward to calculate every cluster gram value
-- igram_cluster_all: It seems the same to counts....I am frastruted.
-- use pairs in gram_cluster_all to get the average gram value

