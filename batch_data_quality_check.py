from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone, timedelta, date

from databricks.sdk.runtime import dbutils, spark

from pyspark.sql import DataFrame

from pyspark.sql.functions import (
    col,
    lit,
    expr,
    current_timestamp,
    from_utc_timestamp,
    when,
    concat,
    to_date,
    max as spark_max
)

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType
)

from pyspark.sql import DataFrame

@dataclass
class ErrorLog:
    ds_job_id: Optional[str] = None
    ds_job_run_id: Optional[str] = None
    ds_job_name: Optional[str] = None
    ds_file_path: Optional[str] = None
    nr_row_count: Optional[int] = None
    ds_catalog: Optional[str] = None
    ds_schema: Optional[str] = None
    ds_table_name: Optional[str] = None
    ds_task_run_id: Optional[date] = None
    ds_check_status: Optional[str] = 'PASS'
    ds_source_system: Optional[str] = None
    cd_error_code: Optional[str] = None
    ds_project: Optional[str] = None
    ds_issues_summary: Optional[str] = None
    ds_remark: Optional[str] = None
    ds_notebook_name: Optional[str] = None
    ds_source_timezone: Optional[str] = None    
    dt_initial_insert_timestamp: Optional[datetime] = None
    dt_insert_timestamp: Optional[datetime] = None
    spark_df: Optional[DataFrame] = None

def get_ops_catalog(log):
    try:
        return dbutils.widgets.get("ops_catalog")
    except Exception:
        return log.ds_catalog
    
def insert_log_table(log):
    ops_catalog = get_ops_catalog(log)
    log_catalog_table = f"{ops_catalog}.manufacturing_staging.raw_app_batch_data_quality_check_log"
    
    schema = StructType([
        StructField("ds_job_id", StringType(), True),
        StructField("ds_job_run_id", StringType(), True),
        StructField("ds_job_name", StringType(), True),
        StructField("ds_catalog", StringType(), True),
        StructField("ds_schema", StringType(), True),
        StructField("ds_table_name", StringType(), True),
        StructField("ds_task_run_id", StringType(), True),
        StructField("ds_check_status", StringType(), True),
        StructField("cd_error_code", StringType(), True),
        StructField("ds_project", StringType(), True),
        StructField("ds_source_system", StringType(), True),
        StructField("ds_file_path", StringType(), True),
        StructField("ds_issues_summary", StringType(), True),
        StructField("ds_remark", StringType(), True),
        StructField("ds_notebook_name", StringType(), True),
        StructField("nr_row_count", IntegerType(), True)
    ])

    log_df = spark.createDataFrame(
        [(
            log.ds_job_id,
            log.ds_job_run_id,
            log.ds_job_name,
            log.ds_catalog,
            log.ds_schema,
            log.ds_table_name,
            str(log.ds_task_run_id) if log.ds_task_run_id else None,
            log.ds_check_status,
            log.cd_error_code,
            log.ds_project,
            log.ds_source_system,
            log.ds_file_path,
            log.ds_issues_summary,
            log.ds_remark,
            log.ds_notebook_name,
            log.nr_row_count
        )],
        schema
    )

    tag_df = (
        spark.table("system.information_schema.table_tags")
        .filter(col("tag_name") == "project")
        .select(
            "catalog_name",
            "schema_name",
            "table_name",
            col("tag_value").alias("project_tag")
        )
    )

    error_df = (
        spark.table(
            f"{ops_catalog}.manufacturing_staging.raw_app_data_quality_check_error_code"
        )
        .select(
            "cd_error_code",
            "ds_error_name",
            "ds_description"
        )
    )

    result_df = (
        log_df.alias("a")
        .join(
            tag_df.alias("b"),
            (
                (col("a.ds_catalog") == col("b.catalog_name"))
                & (col("a.ds_schema") == col("b.schema_name"))
                & (col("a.ds_table_name") == col("b.table_name"))
            ),
            "left"
        )
        .join(
            error_df.alias("c"),
            col("a.cd_error_code") == col("c.cd_error_code"),
            "left"
        )
        .select(
            expr("uuid()").alias("cd_log_id"),
            col("a.ds_job_id"),
            col("a.ds_job_run_id"),
            col("a.ds_job_name"),
            col("a.ds_catalog"),
            col("a.ds_schema"),
            col("a.ds_table_name"),
            col("a.ds_task_run_id"),
            col("a.ds_check_status"),
            col("a.cd_error_code"),
            col("a.nr_row_count"),
            when(
                col("a.ds_project").isNull(),
                col("project_tag")
            ).otherwise(
                col("a.ds_project")
            ).alias("ds_project"),
            col("a.ds_file_path"),
            when(
                col("a.ds_issues_summary").isNull(),
                concat(
                    col("c.ds_error_name"),
                    lit(": "),
                    col("c.ds_description")
                )
            ).otherwise(
                concat(
                    col("c.ds_error_name"),
                    lit(": "),
                    col("a.ds_issues_summary")
                )
            ).alias("ds_issues_summary"),
            col("a.ds_source_system"),
            col("a.ds_remark"),
            col("a.ds_notebook_name"),
            lit("PRC").alias("ds_source_timezone"),
            from_utc_timestamp(
                current_timestamp(),
                "Asia/Shanghai"
            ).alias("dt_insert_timestamp"),
            from_utc_timestamp(
                current_timestamp(),
                "Asia/Shanghai"
            ).alias("dt_initial_insert_timestamp")
        )
    )

    display(result_df)

    result_df.write.mode("append").saveAsTable(log_catalog_table)
	
from threading import Thread

def safe_option(opt):
    try:
        return opt.get() if opt.isDefined() else None
    except:
        return None

def init_log(
        table_name,
        file_path,
        ctx,
        source_system=None):
    log = ErrorLog(
        ds_job_id=safe_option(ctx.jobId()),
        ds_job_name=safe_option(ctx.jobName()),
        ds_file_path=file_path,
        ds_job_run_id=safe_option(ctx.jobRunId()),
        ds_task_run_id=safe_option(ctx.runId()),
        ds_notebook_name=ctx.notebookPath().get().split("/")[-1],
        ds_source_system=source_system,
    )
    parts = table_name.split(".")
    valid_catalogs = {
        "dev_operations",
        "qc_operations",
        "prod_operations"
    }
    if len(parts) < 3:
        log.ds_check_status = "FAILED"
        log.cd_error_code = "060"
        log.ds_issues_summary = (
            "Table name must be in the format "
            "'catalog.schema.table'."
        )
    elif parts[0] not in valid_catalogs:
        log.ds_check_status = "FAILED"
        log.cd_error_code = "060"
        log.ds_issues_summary = (
            f"Invalid catalog '{parts[0]}'. "
            f"Allowed values are: {', '.join(sorted(valid_catalogs))}."
        )
        # For the insert_log_table function, the schema is selected automatically based on the catalog retrieved from the environment. This prevents users from accidentally using an incorrect catalog in the dev environment, where the catalog parameter is not supported.
        log.ds_catalog = 'dev_operations'
    else:
        log.ds_catalog = parts[0]
        log.ds_schema = parts[1]
        log.ds_table_name = parts[2]
    return log


def run_checks(log, checks):
    priority = {
        "check_file_exists": 1,
        "check_schema_strict": 2,
        "check_file_update": 3,
        "check_primary_key": 4
    }
    def get_check_name(check):
        if hasattr(check, "__name__"):
            return check.__name__
        return ""
    checks = sorted(checks,key=lambda x: priority.get(get_check_name(x), 999))
    
    try:
        for check in checks:
            if log.ds_check_status == "FAILED":
                break
            log = check(log)        
    except Exception as e:
        log.ds_check_status = "FAILED"
        log.cd_error_code = "069"
        log.ds_issues_summary = str(e).split('\n')[0]
        print(e)
    finally:
        insert_log_table(log)
    return log

from pyspark.sql.functions import col, to_date, count
    
def load_file(log):
    if log.spark_df is None:
        if isinstance(log.ds_file_path, str):
            log.spark_df = spark.read.parquet(log.ds_file_path)
        elif isinstance(log.ds_file_path, list):
            log.spark_df = spark.read.parquet(*log.ds_file_path)
        else:
            log.cd_error_code = '061'
            log.ds_check_status = 'FAILED'
            log.ds_issues_summary = 'The parameter:file_path must be a string or a list of strings when calling the init_log() function.'
    return log.spark_df

# Check if the file exists and is not empty
def check_file_exists(log):
    spark_df = load_file(log)
    try:
        row_count = spark_df.count()
        log.nr_row_count = row_count
    except Exception as e:
        log.ds_check_status = "FAILED"
        log.cd_error_code = "001"
        log.ds_issues_summary = str(e).split("\n")[0]
        return log
    if row_count == 0:
        log.ds_check_status = "WARNING"
        log.cd_error_code = "002"
        log.ds_issues_summary = (
            f"File {log.ds_file_path} is empty"
        )
    return log

# Check whether the source file is up-to-date
def check_file_update(time_cols, log, tz_offset=8):
    # Validate input parameter
    if not time_cols:
        log.ds_check_status = "FAILED"
        log.cd_error_code = "061"
        log.ds_issues_summary = (
            "The time_cols parameter is required when calling the check_file_update function"
        )
        return log
    # Support both single column and multiple columns
    if isinstance(time_cols, str):
        time_cols = [time_cols]
    spark_df = load_file(log)
    # Verify all specified columns exist in the source file
    missing_cols = [c for c in time_cols if c not in spark_df.columns]
    if missing_cols:
        log.ds_check_status = "FAILED"
        log.cd_error_code = "062"
        log.ds_issues_summary = (
            f"Column(s) {missing_cols} do not exist in the source file when calling the check_file_update function"
        )
        return log
    # Get current date based on the specified timezone
    today = datetime.now(
        timezone(timedelta(hours=tz_offset))
    ).date()
    # Retrieve the latest date for all columns in a single Spark job
    agg_exprs = [
        spark_max(to_date(col(c))).alias(c)
        for c in time_cols
    ]
    max_dates = spark_df.agg(*agg_exprs).collect()[0].asDict()
    invalid_cols = []
    # Check whether each column contains today's date
    for col_name, max_date in max_dates.items():
        if max_date is None:
            invalid_cols.append(
                f"{col_name}(actual=NULL, expected={today})"
            )
        elif max_date < today:
            invalid_cols.append(
                f"{col_name}(actual={max_date}, expected={today})"
            )
    # Fail the check if any column is not up-to-date
    if invalid_cols:
        log.ds_check_status = "WARNING"
        log.cd_error_code = "003"
        log.ds_issues_summary = (
            f"File {log.ds_file_path} is not up-to-date. "
            f"The following date column(s) failed validation: "
            f"{'; '.join(invalid_cols)}"
        )
    return log

# Check if Primary key missing or duplicated
def check_primary_key(pk_cols, log):
    if not pk_cols:
        log.ds_check_status = "FAILED"
        log.cd_error_code = '061'
        log.ds_issues_summary = (
            "The pk_cols parameter is required when calling the check_primary_key function"
        )
    else:
        spark_df = load_file(log)
        # Check if the pk_cols exists in the file schema
        missing_cols = [c for c in pk_cols if c not in spark_df.columns]
        if missing_cols:
            log.ds_check_status = "FAILED"
            log.cd_error_code = '062'
            log.ds_issues_summary = (
                f"Column '{pk_cols}' does not exist in source file when calling the check_primary_key function"
            )
            return log
        null_condition = " OR ".join([f"{c} IS NULL" for c in pk_cols])
        null_cnt = spark_df.filter(null_condition).count()
        if null_cnt > 0:
            log.ds_check_status = "FAILED"
            log.cd_error_code = '010'
            log.ds_issues_summary = (
                f"Primary key contains {null_cnt} rows with NULL values"
            )
            return log
        # Check duplicate primary key
        duplicate_cnt = (
            spark_df
            .groupBy(*pk_cols)
            .count()
            .filter(col("count") > 1)
            .count()
        )
        if duplicate_cnt > 0:
            log.ds_check_status = "FAILED"
            if len(pk_cols) == 1:
                log.cd_error_code = '030'
                log.ds_issues_summary = (f"Found {duplicate_cnt} duplicated primary key(s)")
            else:
                log.cd_error_code = '031'
                log.ds_issues_summary = (f"Found {duplicate_cnt} duplicated union primary key(s)")
    return log

#Check if the schema of the file should match the schema of the table
def check_schema_strict(log):
    file_df = load_file(log)
    table_df = spark.table(log.ds_catalog + "." + log.ds_schema + "." + log.ds_table_name)
    exclude_cols = {
        "dt_initial_insert_timestamp",
        "dt_insert_timestamp",
        "ds_source_timezone"
    }
    file_schema = {
        (f.name.lower(), str(f.dataType))
        for f in file_df.schema.fields
    }
    table_schema = {
        (f.name.lower(), str(f.dataType))
        for f in table_df.schema.fields
        if f.name.lower() not in exclude_cols
    }
    if file_schema != table_schema:
        missing_cols = table_schema - file_schema
        extra_cols = file_schema - table_schema
        log.ds_check_status = "FAILED"
        log.cd_error_code = '004'
        log.ds_issues_summary = (
            f"Schema mismatch. "
            f"Missing columns/type: {list(missing_cols)}; "
            f"Extra columns/type: {list(extra_cols)}"
        )
    return log