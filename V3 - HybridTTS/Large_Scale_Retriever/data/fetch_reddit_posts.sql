-- Just change the name of the subreddit you want to fetch data from

(SELECT id, title, selftext
FROM `fh-bigquery.reddit_posts.2018_08`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext
FROM `fh-bigquery.reddit_posts.2018_07`
WHERE subreddit = 'AskReddit'
AND  score >=  20
)
UNION DISTINCT
(SELECT id, title, selftext
FROM `fh-bigquery.reddit_posts.2018_06`
WHERE subreddit = 'AskReddit'
AND  score >=  20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2018_05`
WHERE subreddit = 'AskReddit'
AND  score >=  20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2018_04`
WHERE subreddit = 'AskReddit'
AND  score >=  20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2018_03`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2018_02`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2018_01`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_12`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_11`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_10`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_09`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_08`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_07`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_06`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_05`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_04`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_03`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_02`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2017_01`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_12`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_11`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_10`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_09`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_08`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_07`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_06`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_05`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_04`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_03`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_02`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)
UNION DISTINCT
(SELECT id, title, selftext 
FROM `fh-bigquery.reddit_posts.2016_01`
WHERE subreddit = 'AskReddit'
AND  score >= 20
)