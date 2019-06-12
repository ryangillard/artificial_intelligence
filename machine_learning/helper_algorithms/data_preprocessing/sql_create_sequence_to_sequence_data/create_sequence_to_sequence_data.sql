WITH
  # Get min and max dates so we can enumerate the range next
  CTE_date_limits AS (
  SELECT
    CAST(MIN(data_date) AS DATE) AS min_data_date,
    CAST(MAX(data_date) AS DATE) AS max_data_date
  FROM
    tbl_raw_data ),
  
  # Expand date range using date bounds
  CTE_date_range AS (
  SELECT
    UNIX_DATE(calendar_date) AS unix_date
  FROM
    UNNEST( GENERATE_DATE_ARRAY((
        SELECT
          min_data_date
        FROM
          CTE_date_limits), (
        SELECT
          max_data_date
        FROM
          CTE_date_limits), INTERVAL 1 DAY) ) AS calendar_date ),

  # Create start and end date ranges for features data with input_seq_len
  CTE_start_end_features_date_range AS (
  SELECT
    ROW_NUMBER() OVER (ORDER BY unix_date) - 1 AS seq_idx,
    unix_date AS seq_start_date,
    LEAD(unix_date, @input_seq_len) OVER (ORDER BY unix_date) AS seq_end_date
  FROM
    CTE_date_range ),

  # Create sequence date ranges for features data
  CTE_seq_features_date_range AS (
  SELECT
    A.seq_idx,
    A.seq_start_date,
    A.seq_end_date,
    B.unix_date
  FROM
    CTE_start_end_features_date_range AS A
  INNER JOIN
    CTE_date_range AS B
  ON
    A.seq_start_date <= B.unix_date
    AND B.unix_date < A.seq_end_date ),

  # Create start and end date ranges for labels data with output_seq_len
  CTE_start_end_labels_date_range AS (
  SELECT
    ROW_NUMBER() OVER (ORDER BY unix_date) - 1 AS seq_idx,
    unix_date AS seq_start_date,
    LEAD(unix_date, @output_seq_len) OVER (ORDER BY unix_date) AS seq_end_date
  FROM
    CTE_date_range ),

  # Create sequence date ranges for labels data
  CTE_seq_labels_date_range AS (
  SELECT
    A.seq_idx,
    A.seq_start_date,
    A.seq_end_date,
    B.unix_date
  FROM
    CTE_start_end_labels_date_range AS A
  INNER JOIN
    CTE_date_range AS B
  ON
    A.seq_start_date <= B.unix_date
    AND B.unix_date < A.seq_end_date ),

  # Create training data matrix of features and labels
  CTE_training_data AS (
  SELECT
    CAST(data_date AS DATE) AS data_date,
    APPROX_QUANTILES(feature1, 100)[OFFSET(50)] AS med_feature1,
	APPROX_QUANTILES(feature2, 100)[OFFSET(50)] AS med_feature2,
	APPROX_QUANTILES(feature3, 100)[OFFSET(50)] AS med_feature3,
	APPROX_QUANTILES(label, 100)[OFFSET(50)] AS med_label,
  FROM
    tbl_raw_data
  GROUP BY
    CAST(data_date AS DATE)),

  # Join features data to features date ranges
  CTE_features AS (
  SELECT
    A.seq_idx,
    A.seq_start_date,
    A.seq_end_date,
    A.unix_date,
    IFNULL(B.med_feature1, 0) AS med_feature1,
    IFNULL(B.med_feature2, 0) AS med_feature2,
    IFNULL(B.med_feature3, 0) AS med_feature3
  FROM
    CTE_seq_features_date_range AS A
  LEFT JOIN
    CTE_training_data AS B
  ON
    A.unix_date = UNIX_DATE(B.data_date)
  WHERE
    A.seq_end_date IS NOT NULL),

  # Aggregate features data into sequences
  CTE_seq_features AS (
  SELECT
    seq_idx,
    seq_start_date,
    seq_end_date,
    STRING_AGG(CAST(unix_date AS STRING), ';'
    ORDER BY
      unix_date) AS unix_date_agg,
    STRING_AGG(CAST(med_feature1 AS STRING), ';'
    ORDER BY
      unix_date) AS med_feature1_agg,
    STRING_AGG(CAST(med_feature2 AS STRING), ';'
    ORDER BY
      unix_date) AS med_feature2_agg,
    STRING_AGG(CAST(med_feature3 AS STRING), ';'
    ORDER BY
      unix_date) AS med_feature3_agg,
  FROM
    CTE_features
  GROUP BY
    seq_idx,
    seq_start_date,
    seq_end_date),

  # Join labels data to labels date ranges
  CTE_labels AS (
  SELECT
    A.seq_idx,
    A.seq_start_date,
    A.seq_end_date,
    A.unix_date,
    IFNULL(B.med_label, 0) AS med_label
  FROM
    CTE_seq_labels_date_range AS A
  LEFT JOIN
    CTE_training_data AS B
  ON
    A.unix_date = UNIX_DATE(B.data_date)
  WHERE
    A.seq_end_date IS NOT NULL),

  # Aggregate labels data into sequences
  CTE_seq_labels AS (
  SELECT
    seq_idx,
    seq_start_date,
    seq_end_date,
    STRING_AGG(CAST(unix_date AS STRING), ';'
    ORDER BY
      unix_date) AS unix_date_agg,
    STRING_AGG(CAST(med_label AS STRING), ';'
    ORDER BY
      unix_date) AS med_label_agg
  FROM
    CTE_labels
  GROUP BY
    seq_idx,
    seq_start_date,
    seq_end_date)

# Join features with labels with horizon in between
SELECT
  A.seq_idx AS feat_seq_idx,
  DATE_FROM_UNIX_DATE(A.seq_start_date) AS feat_seq_start_date,
  DATE_FROM_UNIX_DATE(A.seq_end_date) AS feat_seq_end_date,
  B.seq_idx AS lab_seq_idx,
  DATE_FROM_UNIX_DATE(B.seq_start_date) AS lab_seq_start_date,
  DATE_FROM_UNIX_DATE(B.seq_end_date) AS lab_seq_end_date,
  med_feature1_agg,
  med_feature2_agg,
  med_feature3_agg,
  med_label_agg
FROM
  CTE_seq_features AS A
INNER JOIN
  CTE_seq_labels AS B
ON
  A.seq_end_date + @horizon = B.seq_start_date
ORDER BY
  A.seq_idx