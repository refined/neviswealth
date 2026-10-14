-- Deleting a document archives it: the row stays, but it is hidden from the API and from search.
ALTER TABLE documents ADD COLUMN archived_at timestamptz;
