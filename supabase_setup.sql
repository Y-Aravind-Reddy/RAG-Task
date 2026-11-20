-- Run this once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Embedding size 384 matches sentence-transformers/all-MiniLM-L6-v2.
-- If you change EMBEDDING_MODEL, update vector(384) to the new model's dimension.

create extension if not exists vector;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(384)
);

create or replace function match_documents (
  query_embedding vector(384),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.metadata @> filter
  order by documents.embedding <=> query_embedding;
end;
$$;
