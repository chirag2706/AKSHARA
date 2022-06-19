import { useEffect } from "react";
import { connect } from "react-redux";
import { BrowserRouter, Switch, Route, Link } from "react-router-dom";
import { authAction, LOGIN_FAIL, LOGIN_SUCCESS } from "../actions/authAction";
import Navbar from "../components/Navbar";
import { validateToken } from "../scripts/authenticate";
import Login from "./auth";
import SearchBar from "../components/Searchbar";
import Document from "./Document";
import Project from "./Project";
import Userprofile from "./Userprofile";
import Home from "./Home";

//TODO: connect with store to get auth info

const App = (props) => {
    useEffect(() => {
        async function load() {
            let token = !props.auth.token ? localStorage.getItem("token") : props.auth.token;
            if (token) {
                let status = await validateToken(token);
                if (status === true) {
                    props.dispatch(
                        authAction(LOGIN_SUCCESS, {
                            token: token,
                            isAuthenticated: true,
                            rememberme: true,
                            message: null,
                        })
                    );
                } else {
                    authAction(LOGIN_FAIL, {
                        message: "Token Expired",
                    });
                }
            }
        }

        load();
    }, []);

    return (
        <BrowserRouter>
            <div>
                <Navbar />
                <Switch>
                    <Route path="/login">
                        <Login />
                    </Route>

                    <Route exact path="/documents">
                        <SearchBar />
                    </Route>

                    <Route exact path="/documents/:projectId">
                        <SearchBar />
                    </Route>

                    <Route exact path="/documents/:projectId/:documentId">
                        <Document />
                    </Route>

                    <Route exact path="/projects">
                        <Project />
                    </Route>

                    <Route exact path="/projects/:id">
                        <Project />
                    </Route>

                    <Route path="/profile">
                        <Userprofile />
                    </Route>
                    
                    <Route path="/">
                        <Home />
                    </Route>
                </Switch>
            </div>
        </BrowserRouter>
    );
};

export default connect((state) => ({ auth: state.auth }))(App);
