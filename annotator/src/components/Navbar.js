import { useEffect, useState } from "react";
import { connect } from "react-redux";
import { useHistory } from "react-router-dom";
import { Link } from "react-router-dom";
import { authAction, LOGOUT } from "../actions/authAction";
import Navnotification from "./Navnotification";

const Navbar = (props) => {
    let { auth, dispatch } = props;
    let history = useHistory();
    // let [currentUserAllProjectsList, setCurrentUserAllProjectsList] = useState({
    //     "project-list":[]
    // });

    useEffect(() => {}, [auth]);

    let handleLogout = (e) => {
        e.preventDefault();
        dispatch(authAction(LOGOUT, {}));

        history.push('/');
    };

    // let handleSideNavbarRendering = async () => {
    //     console.log("handleSideNavbarRendering");

    //     let res = await doGetListOfProjectsOfaParticularUser(auth.token);

    //     setCurrentUserAllProjectsList({
    //         ...currentUserAllProjectsList,
    //         "project-list":res.data["project-list"]
    //     });

    // }

    let navbar = (
        <div>
            {/* <SideNavbar /> */}
            <nav className="navbar-expand-lg navbar navbar-dark bg-dark">
                <div className="container-fluid">
                    {/* <button
                        className="navbar-toggler"
                        type="button"
                        data-bs-toggle="collapse"
                        data-bs-target="#sidebarMenu"
                        aria-controls="sidebarMenu"
                        aria-expanded="false"
                        aria-label="Toggle navigation"
                        >
                        <i className="fas fa-bars"></i>
                    </button> */}
                    
                    <Link className="navbar-brand" to='/'>
                        Akshara
                    </Link>
                    <button
                        className="navbar-toggler"
                        type="button"
                        data-bs-toggle="collapse"
                        data-bs-target="#navbarNav"
                        aria-controls="navbarNav"
                        aria-expanded="false"
                        aria-label="Toggle navigation"
                    >
                        <span className="navbar-toggler-icon"></span>
                    </button>
                    <div className="collapse navbar-collapse" id="navbarNav">
                            {!auth.isAuthenticated ? (
                                <ul className="navbar-nav ms-auto">
                                    <li className="nav-item">
                                        <Link to="/login" className="nav-link">
                                            Login
                                        </Link>
                                    </li>
                                </ul>
                            ) : (
                                <>
                                    <ul className="navbar-nav">
                                        <li className="nav-item">
                                            <Link to="/documents" className="nav-link">
                                                Documents
                                            </Link>
                                        </li>
                                    </ul>
                                    <ul className="navbar-nav ms-auto">
                                        <li className="nav-item">
                                            <Navnotification />
                                        </li>
                                        <li className="nav-item">
                                            <Link to="/projects" className="nav-link">
                                                Projects
                                            </Link>
                                        </li>
                                        <li className="nav-item">
                                            <Link to="/profile" className="nav-link">
                                                Profile
                                            </Link>
                                        </li>
                                        <li className="nav-item">
                                            <Link
                                                to="/logout"
                                                className="nav-link"
                                                onClick={(e) => handleLogout(e)}
                                            >
                                                Logout
                                            </Link>
                                        </li>
                                    </ul>
                                </>
                            )}
                            


                    </div>
                </div>
            </nav>

            {/* <div>
                <button className="btn float-end" data-bs-toggle="offcanvas" data-bs-target="#offcanvas">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="bi bi-arrow-right-square-fill" viewBox="0 0 16 16">
                    <path d="M0 14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2H2a2 2 0 0 0-2 2v12zm4.5-6.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5a.5.5 0 0 1 0-1z"/>
                </svg>
                </button>
            </div> */}
            
        </div>
    );

    return navbar;
};

let mapStateToProps = (state) => ({
    auth: state.auth,
});

export default connect(mapStateToProps)(Navbar);
