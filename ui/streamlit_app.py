import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def login(email: str, password: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/auth/login",
            json={"username": email, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        st.session_state["login_error"] = resp.json().get("detail", "Login failed")
        return None
    except requests.exceptions.RequestException as e:
        st.session_state["login_error"] = f"Could not reach API at {API_BASE_URL}: {e}"
        return None


def run_query(token: str, query: str, use_cache: bool, rerank: bool, verify: bool, top_k: int) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/query",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "query": query,
                "use_cache": use_cache,
                "rerank": rerank,
                "verify": verify,
                "top_k": top_k,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Query failed ({resp.status_code}): {resp.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API at {API_BASE_URL}: {e}")
        return None


def get_cache_metrics(token: str) -> dict | None:
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/cache/metrics",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


def render_login():
    st.title("VeriRAG")
    st.caption("RAG with multi-stage verification for legal contract analysis")

    with st.form("login_form"):
        email = st.text_input("Email", value="admin@company.com")
        password = st.text_input("Password", type="password", value="admin123")
        submitted = st.form_submit_button("Log in")

    if submitted:
        result = login(email, password)
        if result:
            st.session_state["token"] = result["access_token"]
            st.session_state["access_level"] = result["access_level"]
            st.session_state["email"] = email
            st.rerun()
        else:
            st.error(st.session_state.get("login_error", "Login failed"))


def render_app():
    token = st.session_state["token"]

    with st.sidebar:
        st.write(f"Logged in as **{st.session_state['email']}**")
        st.write(f"Access level: `{st.session_state['access_level']}`")
        if st.button("Log out"):
            for key in ("token", "access_level", "email"):
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.subheader("Cache metrics")
        metrics = get_cache_metrics(token)
        if metrics:
            st.metric("Hit rate", f"{metrics['hit_rate'] * 100:.1f}%")
            st.metric("Requests", metrics["total_requests"])
            st.metric("Cost saved", f"${metrics['cost_saved']:.4f}")
        else:
            st.caption("Cache metrics unavailable (Redis not connected)")

    st.title("VeriRAG")
    st.caption("Ask a question about the legal contract corpus")

    col1, col2, col3 = st.columns(3)
    use_cache = col1.checkbox("Use cache", value=True)
    rerank = col2.checkbox("Cohere rerank", value=True)
    verify = col3.checkbox("Verify answer", value=True)
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=20, value=5)

    query = st.text_area("Query", placeholder="What are the termination clauses?")

    if st.button("Run query", type="primary") and query.strip():
        with st.spinner("Retrieving, reranking, and verifying..."):
            result = run_query(token, query, use_cache, rerank, verify, top_k)

        if result:
            st.subheader("Answer")
            st.write(result["answer"])

            badge_cols = st.columns(3)
            badge_cols[0].metric("Cache", "Hit" if result["cache_hit"] else "Miss")
            verification = result.get("verification") or {}
            badge_cols[1].metric(
                "Trustworthy", "Yes" if verification.get("is_trustworthy") else "No"
            )
            badge_cols[2].metric(
                "Confidence", f"{verification.get('overall_score', 0):.1f}%"
            )

            st.caption(
                f"Latency: {result.get('latency_ms', 0)} ms - "
                f"Cost: ${result.get('total_cost', 0):.4f}"
            )

            if verification.get("reasoning"):
                with st.expander("Verification reasoning"):
                    st.write(verification["reasoning"])

            sources = result.get("sources") or []
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for src in sources:
                        st.markdown(f"**Contract {src['contract_id']}** - relevance {src['relevance_score']:.3f}")
                        st.text(src["content"][:400])
                        st.divider()


def main():
    st.set_page_config(page_title="VeriRAG", layout="wide")
    if "token" not in st.session_state:
        render_login()
    else:
        render_app()


if __name__ == "__main__":
    main()
